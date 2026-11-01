# Recovered Mac prototype

This test prototype was recovered from the Mac working tree. It requires `MtlCooperativeTensor`, `Int2b`, and `coordinate`, which are absent from both the current refactor and upstream Metal.jl. It cannot be part of the active test suite. The original bytes remain in `recovery/mac-metal-20261101:test/device/intrinsics/tensor.jl`; this copy is formatted only.

On the Apple M5 the capability query succeeds, but compilation reports these undefined imports and the later LLVM inliner aborts. This is an incomplete tensor API experiment, independent of the material renderer.
