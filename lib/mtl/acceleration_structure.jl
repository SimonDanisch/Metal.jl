# Hardware ray tracing: the methods.
#
# `libmtl.jl` is generated from the headers and already declares every
# acceleration-structure class and every property on them — `vertexBuffer`,
# `triangleCount`, `instanceDescriptorBuffer` and the rest are all there. What
# it does not generate is the METHODS: there is no way to make a descriptor, ask
# the device how big a structure needs to be, allocate one, or encode a build.
#
# That is this file, and it is the whole gap. The flow it enables:
#
#     desc  = MTLPrimitiveAccelerationStructureDescriptor()
#     desc.geometryDescriptors = ...
#     sizes = accelerationStructureSizes(dev, desc)
#     accel = alloc_acceleration_structure(dev, sizes.accelerationStructureSize)
#     scratch = MTLBuffer(dev, sizes.buildScratchBufferSize; storage=PrivateStorage)
#     enc = MTLAccelerationStructureCommandEncoder(cmdbuf)
#     build!(enc, accel, desc, scratch)
#     close(enc)

export MTLPrimitiveAccelerationStructureDescriptor,
       MTLAccelerationStructureTriangleGeometryDescriptor,
       MTLInstanceAccelerationStructureDescriptor,
       MTLAccelerationStructureCommandEncoder,
       accelerationStructureSizes, alloc_acceleration_structure,
       build!, refit!, supports_raytracing

"""
    supports_raytracing(dev) -> Bool

Whether `dev` can build and trace acceleration structures in hardware.
"""
supports_raytracing(dev::MTLDevice) =
    @objc [dev::id{MTLDevice} supportsRaytracing]::Bool

# ── Descriptors ───────────────────────────────────────────────────────────────
#
# `descriptor` is a class method returning an autoreleased instance, which is
# the documented way to make any of these — there is no `alloc`/`init` pair.

"""
    MTLAccelerationStructureTriangleGeometryDescriptor()

Triangle geometry for a bottom-level structure. Set `vertexBuffer`,
`vertexStride` and `triangleCount`, and `indexBuffer`/`indexType` if indexed.
"""
MTLAccelerationStructureTriangleGeometryDescriptor() =
    @objc [MTLAccelerationStructureTriangleGeometryDescriptor descriptor]::MTLAccelerationStructureTriangleGeometryDescriptor

"""
    MTLPrimitiveAccelerationStructureDescriptor()

A bottom-level structure: the geometry itself. Set `geometryDescriptors`.
"""
MTLPrimitiveAccelerationStructureDescriptor() =
    @objc [MTLPrimitiveAccelerationStructureDescriptor descriptor]::MTLPrimitiveAccelerationStructureDescriptor

"""
    MTLInstanceAccelerationStructureDescriptor()

A top-level structure: transformed instances of built primitive structures. Set
`instanceDescriptorBuffer`, `instanceCount` and `instancedAccelerationStructures`.
"""
MTLInstanceAccelerationStructureDescriptor() =
    @objc [MTLInstanceAccelerationStructureDescriptor descriptor]::MTLInstanceAccelerationStructureDescriptor

# ── Sizing and allocation ─────────────────────────────────────────────────────

"""
    accelerationStructureSizes(dev, descriptor) -> MTLAccelerationStructureSizes

How large a structure `descriptor` needs, and how much scratch to build it with.

Asked before allocating either. `accelerationStructureSize` is the structure
itself; `buildScratchBufferSize` is temporary and may be released once the build
has completed; `refitScratchBufferSize` is what a later [`refit!`](@ref) needs,
and is zero unless the descriptor asked for `MTLAccelerationStructureUsageRefit`.
"""
accelerationStructureSizes(dev::MTLDevice, desc) =
    @objc [dev::id{MTLDevice} accelerationStructureSizesWithDescriptor:desc::id{MTLAccelerationStructureDescriptor}]::MTLAccelerationStructureSizes

"""
    alloc_acceleration_structure(dev, bytes) -> MTLAccelerationStructure

An empty structure of `bytes`, ready to be built into.

Opaque: the contents are the driver's own layout and there is nothing to map.
Size it from [`accelerationStructureSizes`](@ref).
"""
alloc_acceleration_structure(dev::MTLDevice, bytes::Integer) =
    @objc [dev::id{MTLDevice} newAccelerationStructureWithSize:bytes::NSUInteger]::MTLAccelerationStructure

# ── The encoder ───────────────────────────────────────────────────────────────

"""
    MTLAccelerationStructureCommandEncoder(cmdbuf)

An encoder for acceleration-structure work, alongside compute and blit.

A build is GPU work like any other: encoded into a command buffer, complete when
it completes.
"""
MTLAccelerationStructureCommandEncoder(buf::MTLCommandBuffer) =
    @objc [buf::id{MTLCommandBuffer} accelerationStructureCommandEncoder]::MTLAccelerationStructureCommandEncoder

"""
    build!(enc, accel, descriptor, scratch, offset = 0)

Encode a build of `descriptor` into `accel`, using `scratch` as working memory.
"""
build!(enc::MTLAccelerationStructureCommandEncoder, accel::MTLAccelerationStructure,
       desc, scratch::MTLBuffer, offset::Integer = 0) =
    @objc [enc::id{MTLAccelerationStructureCommandEncoder} buildAccelerationStructure:accel::id{MTLAccelerationStructure} descriptor:desc::id{MTLAccelerationStructureDescriptor} scratchBuffer:scratch::id{MTLBuffer} scratchBufferOffset:offset::NSUInteger]::Nothing

"""
    refit!(enc, src, descriptor, dst, scratch, offset = 0)

Encode a refit of `src` into `dst` for geometry that moved.

Much cheaper than a rebuild, and the reason to hold a structure across frames.
It does not re-cluster, so geometry that moves far enough degrades traversal
until it is rebuilt. `dst` may be `src`, to refit in place.
"""
refit!(enc::MTLAccelerationStructureCommandEncoder, src::MTLAccelerationStructure,
       desc, dst::MTLAccelerationStructure, scratch::MTLBuffer, offset::Integer = 0) =
    @objc [enc::id{MTLAccelerationStructureCommandEncoder} refitAccelerationStructure:src::id{MTLAccelerationStructure} descriptor:desc::id{MTLAccelerationStructureDescriptor} destination:dst::id{MTLAccelerationStructure} scratchBuffer:scratch::id{MTLBuffer} scratchBufferOffset:offset::NSUInteger]::Nothing
