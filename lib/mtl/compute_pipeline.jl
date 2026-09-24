#
# compute pipeline descriptor
#

export MTLComputePipelineDescriptor

# @objcwrapper managed = true MTLComputePipelineDescriptor <: NSObject

function MTLComputePipelineDescriptor()
    return @objc [MTLComputePipelineDescriptor new]::MTLComputePipelineDescriptor
end

export MTLLinkedFunctions

"""
    MTLLinkedFunctions()

An empty set of functions to link into a pipeline.

Assign `.functions` (a `Vector{MTLFunction}` of `[[visible]]` functions) and set
it on a `MTLComputePipelineDescriptor.linkedFunctions`. This is how a kernel
calls something the Metal frontend had to compile — an `intersector<>`, for
instance — that has no AIR symbol a compiler back-end could emit directly.
"""
function MTLLinkedFunctions()
    return @objc [MTLLinkedFunctions new]::MTLLinkedFunctions
end

#
# compute pipeline state
#

export MTLComputePipelineState

# @objcwrapper managed = true MTLComputePipelineState <: NSObject

function MTLComputePipelineState(dev::MTLDevice, fun::MTLFunction)
    err = Ref{id{NSError}}(nil)
    pipeline = @objc [dev::id{MTLDevice} newComputePipelineStateWithFunction:fun::id{MTLFunction}
                                          error:err::Ptr{id{NSError}}]::Union{Nothing,MTLComputePipelineState}
    pipeline === nothing && throw_error(err[])

    return pipeline
end

# Descriptor-based creation. Needed to attach binary archives (`desc.binaryArchives`) and
# to pass `MTLPipelineOption`s such as `MTLPipelineOptionFailOnBinaryArchiveMiss`. Reflection
# is skipped (unreliable on archive hits); pass a null out-param.
function MTLComputePipelineState(dev::MTLDevice, desc::MTLComputePipelineDescriptor;
                                 options::MTLPipelineOption=MTLPipelineOptionNone)
    err = Ref{id{NSError}}(nil)
    pipeline = @objc [dev::id{MTLDevice} newComputePipelineStateWithDescriptor:desc::id{MTLComputePipelineDescriptor}
                                          options:options::MTLPipelineOption
                                          reflection:C_NULL::Ptr{Cvoid}
                                          error:err::Ptr{id{NSError}}]::Union{Nothing,MTLComputePipelineState}
    pipeline === nothing && throw_error(err[])

    return pipeline
end

# ── Visible function tables ───────────────────────────────────────────────────
#
# How a shader calls a function it was not compiled with. The alternative — an
# `extern` declaration linked by NAME — does not work from MSL SOURCE: the
# frontend that `newLibraryWithSource:` runs resolves symbols immediately and
# refuses an unresolved one ("Undefined symbol(s) for architecture 'air64'"). An
# AIR module can carry an unresolved extern, which is how a Julia kernel calls
# into MSL (`register_linked_function!`); a source-compiled MSL kernel cannot, so
# the other direction goes through a TABLE of function pointers and the kernel
# names an INDEX rather than a symbol.
#
# The receiver and the selector share a line in every `@objc` below. Split
# across two — the receiver alone, the selector under it — the macro mis-parses
# and the call fails with a `TypeError` asserting `id{T}` against the very
# wrapper it was handed, which reads like a conversion problem and is not one.

"""
    MTLVisibleFunctionTableDescriptor(n) -> descriptor

A table with room for `n` functions.
"""
function MTLVisibleFunctionTableDescriptor(n::Integer)
    desc = @objc [MTLVisibleFunctionTableDescriptor new]::MTLVisibleFunctionTableDescriptor
    desc.functionCount = UInt64(n)
    return desc
end

"""
    MTLVisibleFunctionTable(pipeline, desc) -> table

The table for `pipeline`, which is what binds it: a function handle is only
meaningful for the pipeline that linked the function, so a table cannot be built
from the device alone.
"""
MTLVisibleFunctionTable(pipeline::MTLComputePipelineState,
                        desc::MTLVisibleFunctionTableDescriptor) =
    @objc [pipeline::id{MTLComputePipelineState} newVisibleFunctionTableWithDescriptor:desc::id{MTLVisibleFunctionTableDescriptor}]::MTLVisibleFunctionTable

"""
    function_handle(pipeline, fun) -> Union{Nothing,MTLFunctionHandle}

The callable handle for `fun` in `pipeline`, or `nothing` when the function was
not linked into it — which is the shape of the mistake worth catching, since a
table entry left unset is a GPU fault rather than an error.
"""
function_handle(pipeline::MTLComputePipelineState, fun::MTLFunction) =
    @objc [pipeline::id{MTLComputePipelineState} functionHandleWithFunction:fun::id{MTLFunction}]::Union{Nothing,MTLFunctionHandle}

"""    set_function!(table, handle, index) — `index` is ZERO-based, as the shader indexes it."""
set_function!(table::MTLVisibleFunctionTable, handle::MTLFunctionHandle, index::Integer) =
    @objc [table::id{MTLVisibleFunctionTable} setFunction:handle::id{MTLFunctionHandle} atIndex:index::NSUInteger]::Nothing
