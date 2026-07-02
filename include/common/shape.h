#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

constexpr size_t kMaxTensorDims = 4;

struct Tensor;
struct Shape {
    uint8_t ndim = 0;
    std::array<size_t, 4> dims{};

    // Build a Shape from an initializer list of dimension sizes.
    static Shape from_dims(std::initializer_list<size_t> dims);
    // Build a Shape from a vector of dimension sizes.
    static Shape from_dims(const std::vector<size_t>& dims);
};

// Product of all dimensions in shape.
size_t numel_from_shape(const Shape& shape);

// Compute row-major strides for shape into strides.
void init_strides(std::array<size_t, 4>& strides, const Shape& shape);

// Compare tensor.shape against an expected dimension vector.
bool shape_equals(const Tensor& tensor, const std::vector<size_t>& expected);
