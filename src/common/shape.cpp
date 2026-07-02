#include "common/shape.h"
#include "common/tensor.h"

// Build a Shape from an initializer list of dimension sizes.
Shape Shape::from_dims(std::initializer_list<size_t> dims) {
    Shape shape;
    assert(dims.size() <= kMaxTensorDims);
    shape.ndim = static_cast<uint8_t>(dims.size());
    size_t i = 0;
    for (size_t d : dims) {
        assert(d > 0 && "dim=0 not supported");
        shape.dims[i++] = d;
    }
    return shape;
}

// Build a Shape from a vector of dimension sizes.
Shape Shape::from_dims(const std::vector<size_t>& dims) {
    Shape shape;
    assert(dims.size() <= kMaxTensorDims);
    shape.ndim = static_cast<uint8_t>(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
        assert(dims[i] > 0 && "dim=0 not supported");
        shape.dims[i] = dims[i];
    }
    return shape;
}

// Product of all dimensions in shape.
size_t numel_from_shape(const Shape& shape) {
    size_t n = 1;
    for (uint8_t i = 0; i < shape.ndim; i++) {
        n *= shape.dims[i];
    }
    return n;
}

// Compute row-major strides for shape into strides.
void init_strides(std::array<size_t, 4>& strides, const Shape& shape) {
    strides.fill(1);
    if (shape.ndim < 2) {
        return;
    }
    size_t stride = 1;
    for (int i = static_cast<int>(shape.ndim) - 2; i >= 0; --i) {
        stride *= shape.dims[i + 1];
        strides[i] = stride;
    }
}

// Compare tensor.shape against an expected dimension vector.
bool shape_equals(const Tensor& tensor, const std::vector<size_t>& expected) {
    if (tensor.ndim != expected.size()) {
        return false;
    }
    for (uint8_t i = 0; i < tensor.ndim; i++) {
        if (tensor.shape[i] != expected[i]) {
            return false;
        }
    }
    return true;
}
