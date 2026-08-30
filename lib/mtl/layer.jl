# The presentation layer: `CAMetalLayer` and its drawables.
#
# This is how anything reaches a screen on Apple platforms. It is QuartzCore
# rather than Metal — a `CAMetalLayer` is a `CALayer` that hands out textures —
# but everything it hands out is a `MTLTexture`, so it belongs beside the rest
# of the Metal surface here.
#
# The model is not a swapchain of images the caller indexes. There is ONE
# drawable at a time: ask for the next, render into its texture, hand it back to
# a command buffer to present. `maximumDrawableCount` bounds how many can be
# outstanding, and `nextDrawable` BLOCKS when they all are — which is the
# frame pacing, and the reason it must not be called until the frame is ready
# to be recorded.

export CAMetalLayer, CAMetalDrawable, next_drawable, present_drawable!

"""A size in points, as Core Graphics spells one."""
struct CGSize
    width::Cdouble
    height::Cdouble
end

@objcwrapper managed = false CAMetalLayer <: NSObject

@objcproperties CAMetalLayer begin
    @autoproperty device::id{MTLDevice} setter = setDevice
    @autoproperty pixelFormat::MTLPixelFormat setter = setPixelFormat
    @autoproperty drawableSize::CGSize setter = setDrawableSize
    @autoproperty framebufferOnly::Bool setter = setFramebufferOnly
    @autoproperty maximumDrawableCount::NSUInteger setter = setMaximumDrawableCount
    @autoproperty displaySyncEnabled::Bool setter = setDisplaySyncEnabled
    @autoproperty contentsScale::Cdouble setter = setContentsScale
end

@objcwrapper managed = false CAMetalDrawable <: MTLDrawable

@objcproperties CAMetalDrawable begin
    @autoproperty texture::id{MTLTexture}
    @autoproperty layer::id{CAMetalLayer}
end

"""
    CAMetalLayer(dev, width, height; format, vsync) -> CAMetalLayer

A layer that hands out textures of `dev`, sized in PIXELS.

`framebufferOnly = false` so the frame can be read back — a layer that is only
ever presented can say `true` and let the driver pick a more compressed layout,
but a demo that also writes a PNG cannot. `drawableSize` is in pixels while a
layer's own `bounds` are in points, which is the whole of the Retina story: a
layer attached to a view has to be told the backing size or it renders at half
resolution and is scaled up.
"""
function CAMetalLayer(dev::MTLDevice, width::Integer, height::Integer;
                      format::MTLPixelFormat = MTLPixelFormatBGRA8Unorm,
                      vsync::Bool = true, readable::Bool = true)
    layer = @objc [CAMetalLayer layer]::id{CAMetalLayer}
    iszero(UInt(layer)) && error("could not create a CAMetalLayer")
    l = CAMetalLayer(layer)
    l.device = dev
    l.pixelFormat = format
    l.drawableSize = CGSize(Cdouble(width), Cdouble(height))
    l.framebufferOnly = !readable
    l.displaySyncEnabled = vsync
    return l
end

"""
    next_drawable(layer) -> CAMetalDrawable or nothing

The next texture to render into, or `nothing` if none became free in time.

`nothing` is a normal answer, not an error: the layer bounds how many drawables
are outstanding and `nextDrawable` gives up rather than waiting for ever. A
caller skips the frame.
"""
function next_drawable(layer::CAMetalLayer)
    d = @objc [layer::id{CAMetalLayer} nextDrawable]::id{CAMetalDrawable}
    return iszero(UInt(d)) ? nothing : CAMetalDrawable(d)
end

"""
    present_drawable!(cmdbuf, drawable)

Show `drawable` when `cmdbuf` completes.

On the command buffer rather than on the drawable, so presentation is ordered
behind the work that drew it. `[drawable present]` exists and is the racy
version — it shows whatever is in the texture at the moment it is called.
"""
present_drawable!(cmdbuf::MTLCommandBuffer, drawable::CAMetalDrawable) =
    @objc [cmdbuf::id{MTLCommandBuffer} presentDrawable:drawable::id{CAMetalDrawable}]::Nothing
