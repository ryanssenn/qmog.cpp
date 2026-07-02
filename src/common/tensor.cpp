#include "common/tensor.h"

#include <cassert>
#include <cstring>

// Build a Tensor view from raw fields.
static Tensor make_tensor(std::shared_ptr<Storage> storage, size_t offset, DType dtype, const Shape& shape) {
    Tensor t;
    t.dtype = dtype;
    t.ndim = shape.ndim;
    t.shape = shape.dims;
    t.numel = numel_from_shape(shape);
    init_strides(t.strides, shape);

    t.storage = std::move(storage);
    t.offset = offset;

    return t;
}

// Wrap existing memory using a pre-built Shape.
Tensor Tensor::from_ptr(void* data, DType dtype, const Shape& shape, Device device) {
    size_t bytes = numel_from_shape(shape) * dtype_size(dtype);
    return make_tensor(make_storage(device, data, bytes), 0, dtype, shape);
}

// View into existing storage at a byte offset.
Tensor Tensor::view(std::shared_ptr<Storage> storage, size_t offset, DType dtype, const Shape& shape) {
    return make_tensor(std::move(storage), offset, dtype, shape);
}

// Allocate fresh storage from arena and return a view into it.
Tensor Tensor::alloc(Arena& arena, DType dtype, std::initializer_list<size_t> dims) {
    Shape shape = Shape::from_dims(dims);
    size_t bytes = numel_from_shape(shape) * dtype_size(dtype);
    return make_tensor(arena.storage, arena.allocate(bytes), dtype, shape);
}

// Offset-adjusted data pointer.
void* Tensor::data() const {
    return static_cast<char*>(storage->cpu_ptr()) + offset;
}

// Typed data pointer; asserts dtype is F32.
float* Tensor::f32() const {
    assert(dtype == DType::F32);
    return static_cast<float*>(data());
}

// Typed data pointer; asserts dtype is F16.
__fp16* Tensor::f16() const {
    assert(dtype == DType::F16);
    return static_cast<__fp16*>(data());
}

// Copy same-dtype elements from src into this tensor.
void Tensor::copy_from(const Tensor& src) {
    assert(dtype == src.dtype);
    assert(numel == src.numel);
    std::memcpy(data(), src.data(), byte_size());
}

// Subtensor view at the given leading indices.
Tensor Tensor::at(std::initializer_list<size_t> idx) const {
    assert(idx.size() <= ndim && "Too many indices for tensor");
    size_t new_offset = offset;

    uint8_t i = 0;
    for (size_t v : idx) {
        assert(v < shape[i] && "Index out of range");
        new_offset += strides[i] * v * type_size();
        i++;
    }

    Shape new_shape;
    new_shape.ndim = ndim - i;
    for (uint8_t j = 0; j < new_shape.ndim; j++) {
        new_shape.dims[j] = shape[i + j];
    }
    return make_tensor(storage, new_offset, dtype, new_shape);
}

// View the same data with a different shape; numel must match.
Tensor Tensor::reshape(std::initializer_list<size_t> new_dims) const {
    Shape new_shape = Shape::from_dims(new_dims);
    size_t new_numel = numel_from_shape(new_shape);
    assert(new_numel == numel && "Reshape size mismatch");
    return make_tensor(storage, offset, dtype, new_shape);
}
