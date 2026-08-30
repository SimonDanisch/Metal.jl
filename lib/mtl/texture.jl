export MTLTextureDescriptor, MTLTexture

## bitwise operations lose type information, so allow conversions
Base.convert(::Type{MTLPixelFormat}, x::Integer) = MTLPixelFormat(x)

function minimumLinearTextureAlignmentForPixelFormat(dev, format)
    return @objc [dev::MTLDevice minimumLinearTextureAlignmentForPixelFormat:format::MTLPixelFormat]::NSUInteger
end

## bitwise operations lose type information, so allow conversions
Base.convert(::Type{MTLTextureUsage}, x::Integer) = MTLTextureUsage(x)

# @objcwrapper managed = true MTLTextureDescriptor <: NSObject

function MTLTextureDescriptor(pixelFormat, width, height, mipmapped=false)
    return @objc [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixelFormat::MTLPixelFormat
                                          width:width::NSUInteger
                                          height:height::NSUInteger
                                          mipmapped:mipmapped::Bool]::MTLTextureDescriptor
end

# @objcwrapper managed = true MTLTexture <: NSObject

function MTLTexture(buffer, descriptor, offset, bytesPerRow)
    return @objc [buffer::id{MTLBuffer} newTextureWithDescriptor:descriptor::id{MTLTextureDescriptor}
                                          offset:offset::NSUInteger
                                          bytesPerRow:bytesPerRow::NSUInteger]::MTLTexture
end

function MTLTexture(dev, descriptor)
    return @objc [dev::id{MTLDevice} newTextureWithDescriptor:descriptor::id{MTLTextureDescriptor}]::MTLTexture
end

"""
    MTLTexture(heap, descriptor, offset)

Place a texture at `offset` in a `MTLHeapTypePlacement` heap.

The texture form of `MTLBuffer(heap, …)`, and the one a frame graph needs: two
render targets whose lifetimes do not overlap are created at the same offset
and share the bytes. `offset` must be a multiple of the alignment
[`heap_texture_size_and_align`](@ref) reported for `descriptor`.

Not the same as `MTLTexture(buffer, descriptor, offset, bytesPerRow)` — that one
is a LINEAR texture over buffer memory, which an Apple GPU cannot use as a
render target.
"""
function MTLTexture(heap::MTLHeap, descriptor, offset)
    ptr = @objc [heap::id{MTLHeap} newTextureWithDescriptor:descriptor::id{MTLTextureDescriptor}
                                   offset:offset::NSUInteger]::id{MTLTexture}
    # Metal signals failure by returning nil, and the two ways to get one here
    # are worth telling apart: the heap is too small, or `offset` does not
    # satisfy the alignment `heap_texture_size_and_align` reported. Adopting the
    # nil instead raises `UndefRefError` from inside ObjectiveC, which names
    # neither.
    if iszero(UInt(ptr))
        sa = heap_texture_size_and_align(heap.device, descriptor)
        error("could not place a $(descriptor.width)×$(descriptor.height) " *
              "$(descriptor.pixelFormat) texture at offset $offset of a " *
              "$(heap.size)-byte heap: it needs $(sa.size) bytes aligned to " *
              "$(sa.align)" *
              (offset % sa.align == 0 ? "" : " — and $offset is not"))
    end
    return MTLTexture(ptr)
end

export getBytes!

"""
    getBytes!(dst, tex, bytesPerRow, region, level = 0)

Copy a texture region into host memory.

The counterpart to `replaceRegion:`. Needed because a render target cannot be a
buffer-backed (linear) texture on an Apple GPU, so reading a rendered image back
has to go through the texture rather than through shared buffer memory.
"""
function getBytes!(dst::Ptr, tex::MTLTexture, bytesPerRow::Integer,
                   region::MTLRegion, level::Integer = 0)
    @objc [tex::id{MTLTexture} getBytes:dst::Ptr{Cvoid}
                               bytesPerRow:bytesPerRow::NSUInteger
                               fromRegion:region::MTLRegion
                               mipmapLevel:level::NSUInteger]::Nothing
end
