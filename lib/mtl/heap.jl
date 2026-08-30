#
# heap descriptor
#

export MTLHeapDescriptor

# @objcwrapper managed = true MTLHeapDescriptor <: NSObject

function MTLHeapDescriptor()
    return @objc [MTLHeapDescriptor new]::MTLHeapDescriptor
end


#
# heap
#

export MTLHeap

# @objcwrapper managed = true MTLHeap <: MTLAllocation

function MTLHeap(dev::MTLDevice, desc::MTLHeapDescriptor)
    return @objc [dev::id{MTLDevice} newHeapWithDescriptor:desc::id{MTLHeapDescriptor}]::MTLHeap
end


"""
    heap_texture_size_and_align(dev, desc) -> MTLSizeAndAlign

How much heap space a texture matching `desc` needs, and what it must be
aligned to.

Asked BEFORE the heap exists, which is the point: a placement heap has to be
big enough for everything that will live in it, and a texture cannot be created
to find out how big it is. The buffer side needs no equivalent because a
buffer's size is its byte count.
"""
function heap_texture_size_and_align(dev, desc)
    return @objc [dev::id{MTLDevice} heapTextureSizeAndAlignWithDescriptor:desc::id{MTLTextureDescriptor}]::MTLSizeAndAlign
end

"""
    makeAliasable!(res)

Say that `res` no longer uses its heap memory, so a resource placed over the
same range may.

On a `MTLHeapTypeAutomatic` heap this is how space is reused at all. A
placement heap aliases by construction — two resources created at overlapping
offsets share bytes whether or not this is called — but calling it keeps the
driver's own tracking honest about which of them is live.
"""
makeAliasable!(res) = @objc [res::id{MTLResource} makeAliasable]::Nothing
