#
# event
#

export MTLEvent

# @objcwrapper managed = true MTLEvent <: NSObject

function MTLEvent(dev::MTLDevice)
    return @objc [dev::id{MTLDevice} newEvent]::MTLEvent
end

"""
    eventid(ev) -> id{MTLEvent}

`ev` as the `id<MTLEvent>` the selectors that signal and wait take, for every kind of
event including an `MTLSharedEvent`.

By `reinterpret`, because passing an `MTLSharedEvent` where `id{MTLEvent}` is declared
goes through ObjectiveC.jl's `convert(id{MTLEvent}, ::id{MTLSharedEvent})`, which checks
the subclass by building `Object{<:MTLEventKind}` at run time: a `TypeVar`, a `DataType`
and their `svec`, 96 bytes measured, on every call the compiler did not fold — and these
are once-a-frame calls on an MTL4 queue, three of them. `ev::MTLEventLike` is the check,
done by dispatch.
"""
eventid(ev::MTLEventLike) = reinterpret(id{MTLEvent}, pointer(ev))


#
# shared event
#

export MTLSharedEvent, MTLSharedEventHandle

# @objcwrapper managed = true MTLSharedEvent <: MTLEvent

function MTLSharedEvent(dev::MTLDevice)
    return @objc [dev::id{MTLDevice} newSharedEvent]::MTLSharedEvent
end

function waitUntilSignaledValue(ev::MTLSharedEvent, value, timeoutMS=typemax(UInt64))
    @objc [ev::id{MTLSharedEvent} waitUntilSignaledValue:value::UInt64
                        timeoutMS:timeoutMS::UInt64]::Bool
end

## shared event handle

# @objcwrapper managed = true MTLSharedEventHandle <: MTLEvent

function MTLSharedEventHandle(ev::MTLSharedEvent)
    return @objc [ev::id{MTLSharedEvent} newSharedEventHandle]::MTLSharedEventHandle
end
