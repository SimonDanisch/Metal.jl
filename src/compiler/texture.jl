# Sampling a bound 2D texture, on Metal.
#
# `KernelInterface.sample_texture_2d(binding, u, v, component)` names a SLOT, not a
# handle, because that is the shape a descriptor set gives on Vulkan: the texture is
# a module-level resource there and a shader reaches it by number. AIR has no such
# thing — a texture is a PARAMETER of the entry function, and a shader body several
# calls deep cannot name one. So the body emits a placeholder standing for "the
# thing bound at binding N", and a pass replaces it with the entry's parameter.
# That works because `finish_ir!` runs AFTER GPUCompiler's optimisation pipeline, by
# which point the whole shader is inlined into the entry and the parameter is in
# scope at the call site.
#
# ── The pointee type, and why a texture argument used to kill the compiler ───
#
# AIR is a TYPED-pointer dialect and the AGX compiler recognises a texture argument by
# its pointee's struct NAME:
#
#   define <4 x float> @TextureCopy(<4 x float>, <2 x float>,
#                                 %struct._texture_2d_t addrspace(1)* nocapture readonly)
#
# LLVM 22 has only opaque pointers, so that type has to be RECONSTRUCTED on the way
# down. `llvm-downgrade --bitcode-version=14.0` infers pointee types from uses, and a
# parameter whose only use is a call to a declared intrinsic gives it nothing to go on
# — so it emits `{} addrspace(1)*`, its placeholder for "unknown". Handed that, AGX
# looks up a texture descriptor for a struct that is not one, gets null, and
# SEGFAULTS: the crash is inside `AGXCompilerCore`, and `MTLRenderPipelineState`
# reports it as `XPC_ERROR_CONNECTION_INTERRUPTED` after retries, naming nothing.
#
# `byref(<ty>)` is what tells the downgrader. It is an ABI attribute LLVM carries on a
# pointer parameter for exactly this purpose — to say what the pointer points AT — and
# the downgrader reads it where it cannot infer:
#
#   %struct._texture_2d_t addrspace(1)* nocapture readonly byref(%struct._texture_2d_t)
#
# The call to the intrinsic gets a bitcast back to `{}`, because the DECLARATION's
# parameters are still unknown-pointee; AGX does not care, since what it reads is the
# entry's own signature and the function type in the `air.fragment` node — and both of
# those now name the struct.
#
# WHERE THAT WAS MEASURED. Not on a Julia shader: on Apple's. `TextureCopy` from
# CoreDisplay's metallib, packed with Metal.jl's own writer, builds a render pipeline
# from its ORIGINAL bytes; parsed by LLVM 22 and written back through the downgrader
# UNTOUCHED, the same function crashes the compiler service. The two disassemblies
# differ in nothing but `%struct._texture_2d_t`/`%struct._sampler_t` becoming `{}` in
# the signature, the call, the declaration and the metadata. Add `byref` and the
# round-tripped module builds again. So the bug was never in the shader this file
# emits, and an attribute on the parameter is the whole of the fix.
#
# An `elementtype(<ty>)` attribute is NOT read by the downgrader (measured: the
# parameter stays `{}` and the attribute is carried through unread), and neither is a
# zero-offset `getelementptr` naming the struct — which is fragile besides, since
# InstCombine folds one away. An attribute survives every pass that follows.
#

# ── The types a stage's signature declares ───────────────────────────────────

"""
    AIRTexture2D{T}

A 2D texture a stage samples, as the pointee of a `Texture2DPtr{T}`. `T` is what a
sample RETURNS — `Float32` for `texture2d<float>` — and not the texture's storage
format, which the hardware converts from.

Zero-sized and never loaded: it exists so the parameter type says what the pointer
is, which is what `retag_stage!` reads to emit `air.texture` instead of the
`air.buffer` GPUCompiler described it as.
"""
struct AIRTexture2D{T} end

"""
    AIRSamplerState

A sampler, as the pointee of a [`SamplerPtr`](@ref). Address space 2 is AIR's
`constant`, which is where a bound sampler lives.
"""
struct AIRSamplerState end

"""A 2D texture parameter: `ptr addrspace(1)`, `texture2d<T, sample>` in MSL."""
const Texture2DPtr{T} = Core.LLVMPtr{AIRTexture2D{T}, 1}

"""A sampler parameter: `ptr addrspace(2)`, `sampler` in MSL."""
const SamplerPtr = Core.LLVMPtr{AIRSamplerState, 2}

"""What AIR calls a texture parameter's type."""
air_texture_type_name(::Type{Texture2DPtr{T}}) where {T} =
    "texture2d<" * air_stage_name(T) * ", sample>"

"""
    AIRSample4f

What `air.sample_texture_2d.v4f32` returns: the four components, and the residency
bit a sparse texture reports.

A struct rather than a tuple because the LLVM type has to be `{ <4 x float>, i8 }`
exactly — `NTuple{4,VecElement{Float32}}` is the `<4 x float>`, and a plain
`NTuple{4,Float32}` would be `[4 x float]`, which is a different type.
"""
struct AIRSample4f
    value::NTuple{4,VecElement{Float32}}
    residency::UInt8
end

# ── The intrinsic ────────────────────────────────────────────────────────────

# air_sample_texture_2d(texture, sampler, u, v)
#
# One filtered sample at normalised coordinates, with no offset, no explicit LOD and
# the default access hint — the operands the reference call in the header passes for
# `tex.sample(s, uv)`.
#
# A comment and not a docstring: `@device_function` expands into a CPU stub and an
# overlay method, and Julia cannot attach one docstring to that pair.
@device_function @inline function air_sample_texture_2d(tex::Texture2DPtr{Float32},
                                                       samp::SamplerPtr,
                                                       u::Float32, v::Float32)
    @typed_ccall("air.sample_texture_2d.v4f32", llvmcall, AIRSample4f,
                 (Texture2DPtr{Float32}, SamplerPtr, NTuple{2,VecElement{Float32}},
                  Bool, NTuple{2,VecElement{Int32}}, Bool, Float32, Float32, Int32),
                 tex, samp, (VecElement(u), VecElement(v)),
                 true, (VecElement(Int32(0)), VecElement(Int32(0))), false,
                 0f0, 0f0, Int32(0))
end

# ── The placeholders, and what replaces them ─────────────────────────────────

"""The name of the placeholder a shader body emits for a bound texture."""
const AIR_TEXTURE_PLACEHOLDER = "air.jl.texture_2d_at_binding"

"""The name of the placeholder a shader body emits for a bound sampler."""
const AIR_SAMPLER_PLACEHOLDER = "air.jl.sampler_at_binding"

# texture_2d_at_binding(binding) / sampler_at_binding(binding)
#
# "The texture bound at `binding`", as a value the shader can hand the intrinsic.
# `binding` counts from ZERO, matching `KernelInterface.sample_texture_2d`, and has
# to be a literal: `lower_texture_bindings!` reads it off the call.
#
# `air.jl.` and not `air.`, because nothing in AIR answers these: they exist only
# between the shader body and the pass below, and a module still holding one when it
# reaches the driver is a bug that pass catches.
#
# Defined through `@eval` so the placeholder names have ONE spelling. `@typed_ccall`
# needs a literal, which is what the interpolation gives it.
for (fname, placeholder, RT) in
        ((:texture_2d_at_binding, AIR_TEXTURE_PLACEHOLDER, :(Texture2DPtr{Float32})),
         (:sampler_at_binding, AIR_SAMPLER_PLACEHOLDER, :SamplerPtr))
    @eval @device_function @inline $fname(binding::Int32) =
        @typed_ccall($placeholder, llvmcall, $RT, (Int32,), binding)
end

"""
    sample_texture_2d(binding, u, v, component) -> Float32

One component of a filtered sample from the 2D texture bound at `binding`.

`binding` and `component` both count from zero, which is
`KernelInterface.sample_texture_2d`'s convention and not this package's — a component
is a vector lane and a binding is an encoder slot, and neither is an index into
anything Julia owns.
"""
@inline function sample_texture_2d(binding::Integer, u::Float32, v::Float32,
                                   component::Integer)
    s = air_sample_texture_2d(texture_2d_at_binding(Int32(binding)),
                              sampler_at_binding(Int32(binding)), u, v)
    return @inbounds s.value[Int(component) + 1].value
end

# ── Which parameters are which ───────────────────────────────────────────────

"""
    texture_binding_parameters(job) -> (kinds, argtypes)

Per Julia argument: `1` where it is a texture parameter, `2` where it is a sampler,
`0` otherwise; and the argument types beside it.

Read off the SIGNATURE rather than from the LLVM types, because `ptr addrspace(1)` is
also what every buffer is and the two are only told apart by what the caller
declared.
"""
function texture_binding_parameters(@nospecialize(job::CompilerJob))
    args = collect(job.source.specTypes.parameters[2:end])
    job.config.params.stage === :mesh || pop!(args)   # the output pointer
    kinds = Int[a <: Texture2DPtr ? 1 : a === SamplerPtr ? 2 : 0 for a in args]
    return (kinds, args)
end

"""
    leading_offset(nparams, nmarkers) -> Int

How many parameters sit in FRONT of the first Julia argument.

GPUCompiler's kernel ABI may prepend one that no Julia argument corresponds to, and
`stage_builtins!` appends one per referenced builtin — so the pad has to be computed
against the whole marker list, which covers the appended ones, and not against the
signature alone. Measuring from the end instead moves every binding one slot along
the moment a shader also reads `frag_coord`.
"""
function leading_offset(nparams::Int, nmarkers::Int)
    offset = nparams - nmarkers
    offset >= 0 ||
        error("$nparams parameters for $nmarkers markers: the marker list cannot be " *
              "longer than the signature it describes")
    return offset
end

"""
    lower_texture_bindings!(job, mod, entry, nmarkers)

Replace every "the texture/sampler at binding N" placeholder with the entry parameter
that holds it.

Parameters are matched to bindings BY ORDER: the n-th texture parameter of the
signature is binding `n - 1`, and likewise for samplers. That is the order the host
binds in, so a caller that declares its textures in the order it binds them needs to
know nothing else.
"""
function lower_texture_bindings!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                 entry::LLVM.Function, nmarkers::Int)
    kinds, _ = texture_binding_parameters(job)
    any(!=(0), kinds) || return nothing
    params = collect(LLVM.parameters(entry))
    offset = leading_offset(length(params), nmarkers)
    textures = LLVM.Value[params[i + offset] for i in eachindex(kinds) if kinds[i] == 1]
    samplers = LLVM.Value[params[i + offset] for i in eachindex(kinds) if kinds[i] == 2]
    name_texture_pointees!(mod, entry, kinds, offset)

    for (name, bound, what) in ((AIR_TEXTURE_PLACEHOLDER, textures, "texture"),
                                (AIR_SAMPLER_PLACEHOLDER, samplers, "sampler"))
        haskey(LLVM.functions(mod), name) || continue
        f = LLVM.functions(mod)[name]
        for use in collect(uses(f))
            call = user(use)
            call isa LLVM.CallInst ||
                error("$name is referenced by something other than a call: $call")
            LLVM.parent(LLVM.parent(call)) === entry || error("""
                a $what binding is named from `$(LLVM.name(LLVM.parent(LLVM.parent(call))))`,
                which is not the stage entry. The replacement is an entry PARAMETER, so
                the call has to have been inlined into the entry — every shader is, by
                the time `finish_ir!` runs. A function this did not reach is one the
                optimiser kept out of line, and sampling from it needs the parameter
                threaded through instead.""")
            b = LLVM.operands(call)[1]
            b isa LLVM.ConstantInt || error("""
                the $what binding of a `sample_texture_2d` call is not a constant. It
                names an encoder slot, and the stage's parameter list is fixed at
                compile time, so the slot has to be known then. Compute the
                coordinates, not the binding.""")
            i = convert(Int, b)
            0 <= i < length(bound) || error("""
                this stage samples the $what at binding $i and was compiled for \
                $(length(bound)). A stage declares which bound textures it reads — \
                see `Mantle.FragmentShader`'s `textures` — and the list has to cover \
                every binding the body names.""")
            replace_uses!(call, bound[i + 1])
            erase!(call)
        end
        @assert isempty(uses(f)) "$name still has uses after replacement"
        erase!(f)
    end
    return nothing
end

"""
    texture_argument_metadata!(arg_infos, kinds, argtypes, nmarkers)

Replace the `air.buffer` description of each texture and sampler parameter with its
own tag.

GPUCompiler described them as buffers, which is what a `ptr addrspace(1)` argument is
to the kernel ABI. Left that way, Metal binds a buffer where the stage expects a
texture.

The LEADING operand of each node is the parameter's POSITION, counted from zero, and
it has to name the parameter that actually holds the texture — the same alignment
`lower_texture_bindings!` uses to find it, from the same marker count.
"""
function texture_argument_metadata!(arg_infos::Vector, kinds::Vector{Int},
                                    @nospecialize(argtypes::Vector), nmarkers::Int)
    offset = leading_offset(length(arg_infos), nmarkers)
    tex = 0
    samp = 0
    for (i, k) in enumerate(kinds)
        k == 0 && continue
        pos = i + offset
        md = Metadata[Metadata(ConstantInt(Int32(pos - 1)))]
        if k == 1
            push!(md, MDString("air.texture"))
            loc = tex; tex += 1
            typename = air_texture_type_name(argtypes[i])
            argname = "texture$loc"
        else
            push!(md, MDString("air.sampler"))
            loc = samp; samp += 1
            typename = "sampler"
            argname = "sampler$loc"
        end
        append!(md, Metadata[MDString("air.location_index"),
                             Metadata(ConstantInt(Int32(loc))),
                             Metadata(ConstantInt(Int32(1)))])
        # `air.sample` says the texture is READ through a sampler rather than with
        # `read()`; a sampler entry carries no access tag at all.
        k == 1 && push!(md, MDString("air.sample"))
        append!(md, Metadata[MDString("air.arg_type_name"), MDString(typename),
                             MDString("air.arg_name"), MDString(argname)])
        arg_infos[pos] = MDNode(md)
    end
    return arg_infos
end

"""
    air_named_struct(mod, name) -> LLVM.StructType

The named, bodyless struct `name`, reused if the context already has one.

Reused and not recreated because a second `StructType("struct._texture_2d_t")` in the
same context is a DIFFERENT type named `struct._texture_2d_t.0`, and a stage with two
textures would then declare two unrelated pointee types.
"""
function air_named_struct(mod::LLVM.Module, name::String)
    ref = LLVM.API.LLVMGetTypeByName2(LLVM.context(mod), name)
    return ref == C_NULL ? LLVM.StructType(name) : LLVM.LLVMType(ref)
end

"""
    name_texture_pointees!(mod, entry, kinds, offset)

Say what each texture and sampler parameter POINTS AT, so the AIR downgrader can
reconstruct the typed pointer the driver reads.

A `byref(%struct._texture_2d_t)` attribute, for the reasons the header measures: it is
the one channel the downgrader reads, and unlike a zero-offset `getelementptr` it
survives the InstCombine that `stage_cleanup!` and `lower_air!` each end in.
"""
function name_texture_pointees!(mod::LLVM.Module, entry::LLVM.Function,
                                kinds::Vector{Int}, offset::Int)
    tex = samp = nothing
    for (i, k) in enumerate(kinds)
        k == 0 && continue
        if k == 1
            tex === nothing && (tex = air_named_struct(mod, "struct._texture_2d_t"))
            T = tex
        else
            samp === nothing && (samp = air_named_struct(mod, "struct._sampler_t"))
            T = samp
        end
        push!(parameter_attributes(entry, i + offset), TypeAttribute("byref", T))
    end
    strip_debuginfo_for_byref!(mod)
    return nothing
end

"""
    strip_debuginfo_for_byref!(mod)

Drop the module's debug info, which is what lets a `byref` on an OPAQUE struct reach
the downgrader.

`byref` is an ABI attribute, and LLVM's verifier rejects one whose type is unsized —
which every AIR handle type is, since `%struct._texture_2d_t` has no body and a bodied
struct in its place puts the AGX crash straight back (measured: `type {}` and
`type { i8 }` both segfault where `type opaque` builds). The verifier runs from
`UpgradeDebugInfo`, and `UpgradeDebugInfo` only verifies a module that declares the
CURRENT debug metadata version — which is why Apple's own shaders, carrying none, go
through untouched. Without the flag it strips the debug info instead, so this does by
hand what the downgrader would otherwise do to the same module a moment later.

The cost is line information for a stage that samples, and it is small: the compile
unit is `emissionKind: NoDebug`, so nothing was going to be emitted from it, and a
graphics stage has no exception reporting to attribute either — `finish_ir!` empties
its throw sites before this runs. Nothing else in the module is touched, so a kernel
that samples no texture keeps everything it had.
"""
function strip_debuginfo_for_byref!(mod::LLVM.Module)
    LLVM.strip_debuginfo!(mod)
    # …and the FLAG with it. `strip_debuginfo!` removes the compile unit and every
    # `!dbg` attachment but leaves `!{i32 2, "Debug Info Version", i32 3}` behind, and
    # that flag alone is what `UpgradeDebugInfo` reads to decide whether to verify.
    md = LLVM.metadata(mod)
    haskey(md, "llvm.module.flags") || return nothing
    flags = md["llvm.module.flags"]
    kept = LLVM.MDNode[]
    for node in collect(LLVM.operands(flags))
        ops = collect(LLVM.operands(node))
        isversion = length(ops) >= 2 && ops[2] isa LLVM.MDString &&
                    string(ops[2]) == "Debug Info Version"
        isversion || push!(kept, node)
    end
    empty!(flags)
    for node in kept
        push!(flags, node)
    end
    return nothing
end
