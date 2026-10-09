# Encoding an indirect command buffer from a kernel.
#
# Read off Apple's compiler rather than guessed: an MSL `compute_command` setter in a
# dynamic library, serialized and parsed back, is one call per setter.
# `cmd.concurrent_dispatch_threadgroups(groups, threads)` is
# `air.concurrent_dispatch_threadgroups_compute_command`, taking the buffer's
# `command_buffer` handle — an 8-byte value the kernel reads out of memory, which the
# host writes as the buffer's `gpuResourceID` — the ZERO-based command index, and the
# two grids as `<3 x i32>`.

export concurrent_dispatch_threadgroups!

@device_function concurrent_dispatch_threadgroups!(
        icb::LLVMPtr{Nothing,AS.Device}, index::UInt32,
        groups::NTuple{3,UInt32}, threads::NTuple{3,UInt32}) =
    @typed_ccall("air.concurrent_dispatch_threadgroups_compute_command", llvmcall, Cvoid,
                 (LLVMPtr{Nothing,AS.Device}, UInt32, NTuple{3,VecElement{UInt32}},
                  NTuple{3,VecElement{UInt32}}),
                 icb, index, map(VecElement, groups), map(VecElement, threads))

@doc """
    concurrent_dispatch_threadgroups!(icb, index, groups, threads)

Give command `index` (ZERO-based) of an indirect command buffer its grid, from the
device. `icb` is the buffer's `gpuResourceID` reinterpreted as a device pointer.

Only the grid changes: the pipeline and the buffers the host encoded into the command
stay. The command processor reads a command when an `execute` reaches its range, so a
command rewritten by a dispatch replayed in the SAME `execute` may already have been
read — measured on an M5, it ran the old grid even with a barrier bit between them.
Rewrite it from an earlier `execute`.
""" concurrent_dispatch_threadgroups!
