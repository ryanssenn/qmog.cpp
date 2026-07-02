#pragma once

#include "common/arena.h"
#include "common/dtype.h"
#include "common/shape.h"
#include "common/storage.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>

// View into memory owned by a Storage: (storage, byte offset). Tensor never
// allocates or frees memory itself; the shared Storage owns the lifetime.
struct Tensor {
    DType dtype = DType::F32;
    size_t numel = 0;
    uint8_t ndim = 0;
    std::array<size_t, 4> shape{};
    std::array<size_t, 4> strides{};

    std::shared_ptr<Storage> storage;
    size_t offset = 0;

    // Wrap existing memory, zero copy on both devices (Metal requires a
    // page-aligned pointer, e.g. from mmap).
    static Tensor from_ptr(void* data, DType dtype, const Shape& shape, Device device = Device::CPU);

    // View into existing storage at a byte offset.
    static Tensor view(std::shared_ptr<Storage> storage, size_t offset, DType dtype, const Shape& shape);

    // Allocate fresh storage from arena and return a view into it.
    static Tensor alloc(Arena& arena, DType dtype, std::initializer_list<size_t> dims);

    // Element size in bytes for this tensor's dtype.
    size_t type_size() const { return dtype_size(dtype); }
    // Total storage size in bytes.
    size_t byte_size() const { return numel * type_size(); }
    // Offset-adjusted data pointer.
    void* data() const;
    // Typed data pointer; asserts dtype is F32.
    float* f32() const;
    // Typed data pointer; asserts dtype is F16.
    __fp16* f16() const;

    // Copy same-dtype elements from src into this tensor.
    void copy_from(const Tensor& src);

    // Subtensor view at the given leading indices.
    Tensor at(std::initializer_list<size_t> idx) const;
    // View the same data with a different shape; numel must match.
    Tensor reshape(std::initializer_list<size_t> new_dims) const;
};
