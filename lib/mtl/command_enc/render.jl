export MTLRenderCommandEncoder, MTLRenderPassDescriptor
export MTLSamplerDescriptor, MTLSamplerState, MTLMeshRenderPipelineDescriptor
export use!, set_front_facing_winding!,
       set_pipeline!, set_depth_stencil_state!, set_vertex_buffer!, set_vertex_bytes!,
       set_fragment_buffer!, set_fragment_bytes!, set_fragment_texture!,
       set_fragment_sampler!,
       set_viewport!, set_scissor!, set_cull_mode!, draw_primitives!, draw_primitives_indirect!,
       draw_indexed_primitives!,
       set_mesh_buffer!, set_mesh_bytes!, set_object_buffer!,
       draw_mesh_threadgroups!

# The rasterisation half of the command interface. `compute.jl` is its
# counterpart and the model for the shape of everything here; the ObjC bindings
# and properties already exist in `libmtl.jl`, so this is only the Julia layer
# that was never written because Metal.jl compiled nothing but compute kernels.

export MTLRenderPipelineDescriptor, MTLDepthStencilDescriptor

MTLRenderPassDescriptor() =
    @objc [MTLRenderPassDescriptor renderPassDescriptor]::MTLRenderPassDescriptor

MTLRenderPipelineDescriptor() =
    @objc [MTLRenderPipelineDescriptor new]::MTLRenderPipelineDescriptor

MTLDepthStencilDescriptor() =
    @objc [MTLDepthStencilDescriptor new]::MTLDepthStencilDescriptor

function MTLRenderCommandEncoder(cmdbuf::MTLCommandBuffer, desc::MTLRenderPassDescriptor)
    @objc [cmdbuf::id{MTLCommandBuffer} renderCommandEncoderWithDescriptor:desc::id{MTLRenderPassDescriptor}]::MTLRenderCommandEncoder
end

function MTLRenderPipelineState(dev::MTLDevice, desc::MTLRenderPipelineDescriptor)
    err = Ref{id{NSError}}(nil)
    state = @objc [dev::id{MTLDevice} newRenderPipelineStateWithDescriptor:desc::id{MTLRenderPipelineDescriptor}
                   error:err::Ptr{id{NSError}}]::Union{Nothing,MTLRenderPipelineState}
    state === nothing && throw_error(err[])
    return state
end

# A mesh pipeline is created from its OWN descriptor type, not from
# `MTLRenderPipelineDescriptor` with a mesh function set on it: the object and
# mesh stages have threadgroup sizes and a payload length that a vertex pipeline
# has nowhere to put.
function MTLRenderPipelineState(dev::MTLDevice, desc::MTLMeshRenderPipelineDescriptor)
    err = Ref{id{NSError}}(nil)
    state = @objc [dev::id{MTLDevice} newRenderPipelineStateWithMeshDescriptor:desc::id{MTLMeshRenderPipelineDescriptor}
                   options:MTLPipelineOptionNone::MTLPipelineOption
                   reflection:C_NULL::Ptr{Nothing}
                   error:err::Ptr{id{NSError}}]::Union{Nothing,MTLRenderPipelineState}
    state === nothing && throw_error(err[])
    return state
end

# A sampler, and the descriptor it is built from. Neither was wrapped: nothing in
# Metal.jl sampled a texture until Mantle's overlay path did.
MTLSamplerDescriptor() = @objc [MTLSamplerDescriptor new]::MTLSamplerDescriptor

MTLSamplerState(dev::MTLDevice, desc::MTLSamplerDescriptor) =
    @objc [dev::id{MTLDevice} newSamplerStateWithDescriptor:desc::id{MTLSamplerDescriptor}]::MTLSamplerState

MTLMeshRenderPipelineDescriptor() =
    @objc [MTLMeshRenderPipelineDescriptor new]::MTLMeshRenderPipelineDescriptor

function MTLDepthStencilState(dev::MTLDevice, desc::MTLDepthStencilDescriptor)
    @objc [dev::id{MTLDevice} newDepthStencilStateWithDescriptor:desc::id{MTLDepthStencilDescriptor}]::MTLDepthStencilState
end

set_pipeline!(rce::MTLRenderCommandEncoder, pip::MTLRenderPipelineState) =
    @objc [rce::id{MTLRenderCommandEncoder} setRenderPipelineState:pip::id{MTLRenderPipelineState}]::Nothing

set_depth_stencil_state!(rce::MTLRenderCommandEncoder, st::MTLDepthStencilState) =
    @objc [rce::id{MTLRenderCommandEncoder} setDepthStencilState:st::id{MTLDepthStencilState}]::Nothing

# `index - 1` throughout, matching `compute.jl`: Julia counts buffer slots from
# one, Metal from zero, and the conversion belongs at the boundary rather than
# in every caller.
"""
    set_front_facing_winding!(rce, winding)

Which triangle winding counts as front-facing.

Needed by anything that flips a clip axis: mirroring y reverses the handedness
of every triangle, so what was counter-clockwise on screen becomes clockwise and
back-face culling starts keeping exactly the faces it used to drop.
"""
set_front_facing_winding!(rce::MTLRenderCommandEncoder, w::MTLWinding) =
    @objc [rce::id{MTLRenderCommandEncoder} setFrontFacingWinding:w::MTLWinding]::Nothing

set_vertex_buffer!(rce::MTLRenderCommandEncoder, buf::MTLBuffer, offset, index) =
    @objc [rce::id{MTLRenderCommandEncoder} setVertexBuffer:buf::id{MTLBuffer}
                                            offset:offset::NSUInteger
                                            atIndex:(index-1)::NSUInteger]::Nothing

# The mesh pipeline's buffer slots. Its own family, because a mesh stage's
# arguments are bound separately from a vertex stage's — the two never coexist,
# but the selectors are distinct and Metal binds nothing if the wrong one is used.
set_mesh_buffer!(rce::MTLRenderCommandEncoder, buf::MTLBuffer, offset, index) =
    @objc [rce::id{MTLRenderCommandEncoder} setMeshBuffer:buf::id{MTLBuffer}
                                            offset:offset::NSUInteger
                                            atIndex:(index-1)::NSUInteger]::Nothing

set_mesh_bytes!(rce::MTLRenderCommandEncoder, ptr::Ptr, len::Integer, index::Integer) =
    @objc [rce::id{MTLRenderCommandEncoder} setMeshBytes:ptr::Ptr{Cvoid}
                                            length:len::NSUInteger
                                            atIndex:(index-1)::NSUInteger]::Nothing

set_object_buffer!(rce::MTLRenderCommandEncoder, buf::MTLBuffer, offset, index) =
    @objc [rce::id{MTLRenderCommandEncoder} setObjectBuffer:buf::id{MTLBuffer}
                                            offset:offset::NSUInteger
                                            atIndex:(index-1)::NSUInteger]::Nothing

set_vertex_bytes!(rce::MTLRenderCommandEncoder, ptr::Ptr, len::Integer, index::Integer) =
    @objc [rce::id{MTLRenderCommandEncoder} setVertexBytes:ptr::Ptr{Cvoid}
                                            length:len::NSUInteger
                                            atIndex:(index-1)::NSUInteger]::Nothing

set_fragment_buffer!(rce::MTLRenderCommandEncoder, buf::MTLBuffer, offset, index) =
    @objc [rce::id{MTLRenderCommandEncoder} setFragmentBuffer:buf::id{MTLBuffer}
                                            offset:offset::NSUInteger
                                            atIndex:(index-1)::NSUInteger]::Nothing

set_fragment_bytes!(rce::MTLRenderCommandEncoder, ptr::Ptr, len::Integer, index::Integer) =
    @objc [rce::id{MTLRenderCommandEncoder} setFragmentBytes:ptr::Ptr{Cvoid}
                                            length:len::NSUInteger
                                            atIndex:(index-1)::NSUInteger]::Nothing

# Its counterpart: a sampled texture needs both, and on Metal they occupy two
# separate slot namespaces rather than one combined descriptor.
set_fragment_sampler!(rce::MTLRenderCommandEncoder, st::MTLSamplerState, index) =
    @objc [rce::id{MTLRenderCommandEncoder} setFragmentSamplerState:st::id{MTLSamplerState}
                                            atIndex:(index-1)::NSUInteger]::Nothing

set_fragment_texture!(rce::MTLRenderCommandEncoder, tex::MTLTexture, index) =
    @objc [rce::id{MTLRenderCommandEncoder} setFragmentTexture:tex::id{MTLTexture}
                                            atIndex:(index-1)::NSUInteger]::Nothing

set_viewport!(rce::MTLRenderCommandEncoder, vp::MTLViewport) =
    @objc [rce::id{MTLRenderCommandEncoder} setViewport:vp::MTLViewport]::Nothing

set_scissor!(rce::MTLRenderCommandEncoder, r::MTLScissorRect) =
    @objc [rce::id{MTLRenderCommandEncoder} setScissorRect:r::MTLScissorRect]::Nothing

set_cull_mode!(rce::MTLRenderCommandEncoder, mode::MTLCullMode) =
    @objc [rce::id{MTLRenderCommandEncoder} setCullMode:mode::MTLCullMode]::Nothing

function draw_primitives!(rce::MTLRenderCommandEncoder, prim::MTLPrimitiveType,
                          first::Integer, count::Integer, instances::Integer = 1)
    @objc [rce::id{MTLRenderCommandEncoder} drawPrimitives:prim::MTLPrimitiveType
                                            vertexStart:first::NSUInteger
                                            vertexCount:count::NSUInteger
                                            instanceCount:instances::NSUInteger]::Nothing
end

# A mesh pipeline's draw. There is no vertex count and no primitive type: the
# mesh stage WRITES the primitives, and its `MeshConfig` already said what they
# are, so all the host supplies is how many threadgroups to run and how wide each
# stage's threadgroup is.
#
# `object` is the object stage's threadgroup size and is ignored by a pipeline
# without one, but Metal still requires a valid size; one thread is the
# smallest.
function draw_mesh_threadgroups!(rce::MTLRenderCommandEncoder, groups::MTLSize,
                                 object::MTLSize, mesh::MTLSize)
    @objc [rce::id{MTLRenderCommandEncoder} drawMeshThreadgroups:groups::MTLSize
                                            threadsPerObjectThreadgroup:object::MTLSize
                                            threadsPerMeshThreadgroup:mesh::MTLSize]::Nothing
end

# The counterpart to a compute indirect dispatch: the GPU reads the draw
# arguments out of a buffer, so a count produced on the device never has to
# reach the host.
function draw_primitives_indirect!(rce::MTLRenderCommandEncoder, prim::MTLPrimitiveType,
                                   buf::MTLBuffer, offset::Integer = 0)
    @objc [rce::id{MTLRenderCommandEncoder} drawPrimitives:prim::MTLPrimitiveType
                                            indirectBuffer:buf::id{MTLBuffer}
                                            indirectBufferOffset:offset::NSUInteger]::Nothing
end

function draw_indexed_primitives!(rce::MTLRenderCommandEncoder, prim::MTLPrimitiveType,
                                  count::Integer, itype::MTLIndexType,
                                  ibuf::MTLBuffer, ioffset::Integer,
                                  instances::Integer = 1)
    @objc [rce::id{MTLRenderCommandEncoder} drawIndexedPrimitives:prim::MTLPrimitiveType
                                            indexCount:count::NSUInteger
                                            indexType:itype::MTLIndexType
                                            indexBuffer:ibuf::id{MTLBuffer}
                                            indexBufferOffset:ioffset::NSUInteger
                                            instanceCount:instances::NSUInteger]::Nothing
end

# ── attachment arrays ────────────────────────────────────────────────────────
#
# `desc.colorAttachments` is an ObjC array-like object addressed with
# `objectAtIndexedSubscript:`, not an NSArray, so nothing generic reaches it.
# Both the pipeline's and the pass's arrays need it, and without them there is
# no way to name a colour target at all.
#
# One-based, like the buffer slots above.
# `@objc` type-ASSERTS its receiver rather than converting it, and a property
# getter hands back the wrapper struct, so the `id` has to be produced here.
function Base.getindex(arr::MTLRenderPipelineColorAttachmentDescriptorArray, i::Integer)
    h = Base.unsafe_convert(id{MTLRenderPipelineColorAttachmentDescriptorArray}, arr)
    @objc [h::id{MTLRenderPipelineColorAttachmentDescriptorArray} objectAtIndexedSubscript:(i-1)::NSUInteger]::MTLRenderPipelineColorAttachmentDescriptor
end

function Base.getindex(arr::MTLRenderPassColorAttachmentDescriptorArray, i::Integer)
    h = Base.unsafe_convert(id{MTLRenderPassColorAttachmentDescriptorArray}, arr)
    @objc [h::id{MTLRenderPassColorAttachmentDescriptorArray} objectAtIndexedSubscript:(i-1)::NSUInteger]::MTLRenderPassColorAttachmentDescriptor
end


#### use

"""
    use!(rce, buf, mode = ReadUsage, stages = MTLRenderStageVertex | MTLRenderStageFragment)

Make `buf` resident for this render pass.

Needed whenever a shader reaches memory through an ADDRESS rather than through a
bound slot — which is what a `MtlDeviceArray` argument is, a struct holding a
`gpuAddress`. Metal only maps what an encoder was told about, so an address the
encoder never saw reads as zeros: silently, and only when the page happens not
to be mapped.

The compute counterpart is `use!(::MTLComputeCommandEncoder, …)`, and the
`Adaptor` calls it automatically inside a launch. A render encoder has no such
hook, so the draw path calls this itself.
"""
function use!(rce::MTLRenderCommandEncoder, buf::MTLBuffer,
              mode::MTLResourceUsage = ReadUsage,
              stages::MTLRenderStages = MTLRenderStages(MTLRenderStageVertex |
                                                        MTLRenderStageFragment))
    @objc [rce::id{MTLRenderCommandEncoder} useResource:buf::id{MTLBuffer}
                                            usage:mode::MTLResourceUsage
                                            stages:stages::MTLRenderStages]::Nothing
end
