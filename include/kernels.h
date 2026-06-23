#pragma once

#include "tensor.h"
#include "fp16.h"
#include <cmath>

template <typename WeightT, typename ActivationT, typename AccumT>
void matmul(Tensor<AccumT>& xout, Tensor<WeightT>& w, Tensor<ActivationT>& x);
void row_matmul(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& w);
void row_matmul(Tensor<float>& xout, Tensor<float>& x, Tensor<fp16_t>& w);

void softmax(Tensor<float>& xout, Tensor<float>& x);
void rope(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& cos, Tensor<float>& sin);

void silu(Tensor<float>& xout, Tensor<float>& x);

void add(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& y);

void mul(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& y);
void mul(Tensor<float>& xout, Tensor<float>& x, float c);
