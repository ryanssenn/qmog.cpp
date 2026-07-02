#include "common/storage.h"

#include "backend/metal/metal_backend.h"

std::shared_ptr<Storage> make_storage(Device device, void* ptr, size_t size) {
    switch (device) {
        case Device::CPU:
            return ptr ? std::make_shared<CPUStorage>(ptr)
                       : std::make_shared<CPUStorage>(size);
        case Device::Metal:
            return ptr ? MetalBackend::allocate_buffer(ptr, size)
                       : MetalBackend::allocate_buffer(size);
    }
}
