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
@generated function emit_threadgroup_memory(::Type{T}, ::Val{len} = Val(0),
                                            ::Val{id} = Val(0)) where {T, len, id}
    Context() do ctx
        # XXX: as long as LLVMPtr is emitted as i8*, it doesn't make sense to type the GV
        eltyp = convert(LLVMType, LLVM.Int8Type())
        T_ptr = convert(LLVMType, Core.LLVMPtr{T,AS.ThreadGroup})

        # create a function
        llvm_f, _ = create_function(T_ptr)

        # create the global variable
        mod = LLVM.parent(llvm_f)
        gv_typ = LLVM.ArrayType(eltyp, len * sizeof(T))
        gv = GlobalVariable(mod, gv_typ, "threadgroup_memory_$id", AS.ThreadGroup)
        if len > 0
            linkage!(gv, LLVM.API.LLVMInternalLinkage)
            initializer!(gv, UndefValue(gv_typ))
        end
        alignment!(gv, Base.datatype_alignment(T))

        # generate IR
        IRBuilder() do builder
            entry = BasicBlock(llvm_f, "entry")
            position!(builder, entry)

            ptr = gep!(builder, gv_typ, gv, [ConstantInt(0), ConstantInt(0)])

            untyped_ptr = bitcast!(builder, ptr, T_ptr)

            ret!(builder, untyped_ptr)
        end

        call_function(llvm_f, Core.LLVMPtr{T,AS.ThreadGroup})
    end
end
