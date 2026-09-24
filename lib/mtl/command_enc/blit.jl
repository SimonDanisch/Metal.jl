export MTLBlitCommandEncoder, append_copy!, append_fillbuffer!, append_sync!

# @objcwrapper MTLBlitCommandEncoder <: MTLCommandEncoder

function MTLBlitCommandEncoder(cmdbuf::MTLCommandBuffer)
    @objc [cmdbuf::id{MTLCommandBuffer} blitCommandEncoder]::MTLBlitCommandEncoder
end

## encode in the Command Encoder
function MTLBlitCommandEncoder(f::Base.Callable, cmdbuf::MTLCommandBuffer)
    encoder = MTLBlitCommandEncoder(cmdbuf)
    f(encoder)
    close(encoder)
    return encoder
end

##
# Copy from device to device
function append_copy!(enc::MTLBlitCommandEncoder, dst::MTLBuffer, doff,
                      src::MTLBuffer, soff, len)
    @objc [enc::id{MTLBlitCommandEncoder} copyFromBuffer:src::id{MTLBuffer}
                                          sourceOffset:soff::Csize_t
                                          toBuffer:dst::id{MTLBuffer}
                                          destinationOffset:doff::Csize_t
                                          size:len::Csize_t]::Nothing
end

for T in (UInt8, Int8)
    @eval begin
        function append_fillbuffer!(enc::MTLBlitCommandEncoder, src::MTLBuffer,
                                    val::$T, bytesize, offset=0)
            range = NSRange(offset, bytesize)
            @objc [enc::id{MTLBlitCommandEncoder} fillBuffer:src::id{MTLBuffer}
                                                  range:range::NSRange
                                                  value:val::$T]::Nothing
            end
    end
end

"""
    append_copy!(enc, dst::MTLBuffer, doff, bytes_per_row, bytes_per_image,
                 src::MTLTexture, origin, size, slice = 0, level = 0)

Copy a texture region into buffer memory.

The only way off a PRIVATE texture: `getBytes!` needs shared or managed
storage, and a render target the GPU writes wants neither — a frame graph's
attachments live in device memory and are read back, if at all, through a copy
like this one.

`bytes_per_image` may be 0 for a 2D copy; Metal then derives it from the row
stride and the height.
"""
function append_copy!(enc::MTLBlitCommandEncoder, dst::MTLBuffer, doff::Integer,
                      bytes_per_row::Integer, bytes_per_image::Integer,
                      src::MTLTexture, origin::MTLOrigin, size::MTLSize,
                      slice::Integer = 0, level::Integer = 0)
    @objc [enc::id{MTLBlitCommandEncoder} copyFromTexture:src::id{MTLTexture}
                                          sourceSlice:slice::NSUInteger
                                          sourceLevel:level::NSUInteger
                                          sourceOrigin:origin::MTLOrigin
                                          sourceSize:size::MTLSize
                                          toBuffer:dst::id{MTLBuffer}
                                          destinationOffset:doff::NSUInteger
                                          destinationBytesPerRow:bytes_per_row::NSUInteger
                                          destinationBytesPerImage:bytes_per_image::NSUInteger]::Nothing
end

"""
    append_copy!(enc, dst::MTLTexture, origin, size, src::MTLBuffer, soff,
                 bytes_per_row, bytes_per_image, slice = 0, level = 0)

Copy buffer memory into a texture region: the mirror of the method above.

The way ONTO a texture that is ordered with the rest of a queue. `replaceRegion`
writes from the CPU the moment it is called, while a command buffer that samples
the texture may still be running; a blit is encoded like any other command and
waits its turn. It is also the only way onto a PRIVATE texture, and the way to
fill one from data that is already on the device without a host round trip.

`bytes_per_image` may be 0 for a 2D copy.
"""
function append_copy!(enc::MTLBlitCommandEncoder, dst::MTLTexture, origin::MTLOrigin,
                      size::MTLSize, src::MTLBuffer, soff::Integer,
                      bytes_per_row::Integer, bytes_per_image::Integer,
                      slice::Integer = 0, level::Integer = 0)
    @objc [enc::id{MTLBlitCommandEncoder} copyFromBuffer:src::id{MTLBuffer}
                                          sourceOffset:soff::NSUInteger
                                          sourceBytesPerRow:bytes_per_row::NSUInteger
                                          sourceBytesPerImage:bytes_per_image::NSUInteger
                                          sourceSize:size::MTLSize
                                          toTexture:dst::id{MTLTexture}
                                          destinationSlice:slice::NSUInteger
                                          destinationLevel:level::NSUInteger
                                          destinationOrigin:origin::MTLOrigin]::Nothing
end

# only for managed resources
function append_sync!(enc::MTLBlitCommandEncoder, src::MTLBuffer)
    @objc [enc::id{MTLBlitCommandEncoder} synchronizeResource:src::id{MTLBuffer}]::Nothing
end
