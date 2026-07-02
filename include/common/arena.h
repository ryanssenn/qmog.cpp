#pragma once

#include "common/storage.h"

#include <cassert>
#include <cstddef>
#include <memory>

// Bump allocator for activation buffers. Owns one Storage; allocate() returns
// byte offsets into it. Tensors share the storage and view it at an offset.
struct Arena {
    static constexpr size_t kAlignment = 64;

    std::shared_ptr<Storage> storage;
    size_t capacity;
    size_t offset = 0;

    explicit Arena(size_t size, Device device = Device::CPU)
        : storage(make_storage(device, nullptr, size)), capacity(size) {}

    size_t allocate(size_t size) {
        size_t aligned = (offset + kAlignment - 1) & ~(kAlignment - 1);
        assert(aligned + size <= capacity && "Arena out of memory");
        offset = aligned + size;
        return aligned;
    }
};
