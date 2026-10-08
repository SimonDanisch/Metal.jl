export MtlThreadGroupArray

"""
    MtlThreadGroupArray(::Type{T}, dims, [::Val{id}])

Create an array local to each threadgroup launched during kernel execution.

`id` names the backing global, `threadgroup_memory_\$id`, and keys the generator along
with `T` and the length. Two arrays raised under the default `Val(0)` still come out
distinct — the linker renames the colliding global — but they are distinct by accident
of that rename rather than by construction. Give each allocation its own `id` when a
kernel wants several, so the IR says which is which.
"""
@inline function MtlThreadGroupArray(::Type{T}, dims, id::Val = Val(0)) where {T}
    len = prod(dims)
    # NOTE: this relies on const-prop to forward the literal length to the generator.
    #       maybe we should include the size in the type, like StaticArrays does?
    ptr = emit_threadgroup_memory(T, Val(len), id)
    MtlDeviceArray(dims, ptr)
end

# get a pointer to threadgroup memory, with known (static) or zero length (dynamic)
@llvmgenerated builder function emit_threadgroup_memory(::Type{T}, ::Val{len}=Val(0),
                                                        ::Val{id}=Val(0)
                                                        )::Core.LLVMPtr{T,AS.ThreadGroup} where {T,len,id}
    # XXX: as long as LLVMPtr is emitted as i8*, it doesn't make sense to type the GV
    eltyp = LLVM.Int8Type()
    T_ptr = convert(LLVMType, Core.LLVMPtr{T,AS.ThreadGroup})

    # create the global variable. align and pad it to 4 bytes, so that GPUCompiler can
    # implement 8- and 16-bit atomics on the containing 32-bit word.
    gv_typ = LLVM.ArrayType(eltyp, cld(len * sizeof(T), 4) * 4)
    gv = GlobalVariable(current_module(builder), gv_typ, "threadgroup_memory_$id", AS.ThreadGroup)
    if len > 0
        gv.linkage = LLVM.Linkage.Internal
        gv.initializer = UndefValue(gv_typ)
    end
    gv.alignment = max(Base.datatype_alignment(T), 4)

    ptr = gep!(builder, gv_typ, gv, [ConstantInt(0), ConstantInt(0)])
    bitcast!(builder, ptr, T_ptr)
end
