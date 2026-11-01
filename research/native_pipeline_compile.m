// Compile-only Metal reproducer for an emitted .metallib and entry-name file.
// Build: clang -fobjc-arc -framework Metal -framework Foundation native_pipeline_compile.m -o native_pipeline_compile
// Arguments: library.metallib entry.txt indirect(0|1) [max_threads]
// Use an external process deadline for pathological compiler inputs.
// Does not dispatch the kernel or validate rendered images.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 4 && argc != 5) return 2;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSData *bytes = [NSData dataWithContentsOfFile:@(argv[1])];
        NSString *entry = [[NSString stringWithContentsOfFile:@(argv[2])
            encoding:NSUTF8StringEncoding error:nil]
            stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        if (!device || !bytes || !entry) return 3;
        dispatch_data_t data = dispatch_data_create(bytes.bytes, bytes.length,
            dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
        if (!library) { fprintf(stderr, "%s\n", error.description.UTF8String); return 4; }
        id<MTLFunction> function = [library newFunctionWithName:entry];
        if (!function) return 5;
        MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = function;
        descriptor.supportIndirectCommandBuffers = atoi(argv[3]) != 0;
        if (argc == 5) descriptor.maxTotalThreadsPerThreadgroup = atoi(argv[4]);
        double started = [NSDate timeIntervalSinceReferenceDate];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:descriptor
            options:MTLPipelineOptionNone reflection:nil error:&error];
        if (!pipeline) { fprintf(stderr, "%s\n", error.description.UTF8String); return 6; }
        printf("NATIVE_COMPILE_SUCCESS seconds=%.6f maxthreads=%lu\n",
            [NSDate timeIntervalSinceReferenceDate] - started, pipeline.maxTotalThreadsPerThreadgroup);
    }
    return 0;
}
