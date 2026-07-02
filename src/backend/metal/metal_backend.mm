#include "backend/metal/metal_backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdexcept>
#include <unordered_map>
#include <unistd.h>


struct MetalStorage : public Storage {
    id<MTLBuffer> data;
    size_t size;

    MetalStorage(id<MTLBuffer> data, size_t size)
        : Storage(Device::Metal), data(data), size(size) {}

    void* cpu_ptr() override {
        return [data contents];
    }
};


struct MetalBackend::Impl {
    static id<MTLDevice> device;
    static id<MTLCommandQueue> queue;
    static id<MTLLibrary> library;

    static std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

    static void init() {
        if (device) {
            return;
        }

        device = MTLCreateSystemDefaultDevice();

        if (!device) {
            throw std::runtime_error("No metal device found");
        }

        queue = [device newCommandQueue];

        if (!queue){
            throw std::runtime_error("Failed to create command queue");
        }
    }

    static std::shared_ptr<Storage> allocate_buffer(size_t size){
        id<MTLBuffer> buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];

        return std::make_shared<MetalStorage>(buffer, size);
    }

    // Zero-copy wrap of existing page-aligned memory (mmap'd weights).
    static std::shared_ptr<Storage> allocate_buffer(void* ptr, size_t size){
        size_t page = static_cast<size_t>(getpagesize());
        size_t rounded = (size + page - 1) & ~(page - 1);
        id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:ptr
                                                         length:rounded
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        if (!buffer) {
            throw std::runtime_error("newBufferWithBytesNoCopy failed (pointer not page-aligned?)");
        }

        return std::make_shared<MetalStorage>(buffer, size);
    }


    static void dispatch(const std::string& kernel_name, std::vector<id<MTLBuffer>> buffers) {}

    private:
    static void pipeline_for(const std::string& kernel_name) {}
};

id<MTLDevice> MetalBackend::Impl::device = nil;
id<MTLCommandQueue> MetalBackend::Impl::queue = nil;
id<MTLLibrary> MetalBackend::Impl::library = nil;
std::unordered_map<std::string, id<MTLComputePipelineState>> MetalBackend::Impl::pipelines;

void MetalBackend::init(){
    Impl::init();
}

std::shared_ptr<Storage> MetalBackend::allocate_buffer(size_t size){
    return Impl::allocate_buffer(size);
}

std::shared_ptr<Storage> MetalBackend::allocate_buffer(void* ptr, size_t size){
    return Impl::allocate_buffer(ptr, size);
}
