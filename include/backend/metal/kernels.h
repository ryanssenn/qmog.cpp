#pragma once

#include "common/tensor.h"

void matmul(Tensor& xout, const Tensor& w, const Tensor& x, size_t m = 0);
void row_matmul(Tensor& xout, const Tensor& x, const Tensor& w, size_t n = 0);

void softmax(Tensor& xout, const Tensor& x, size_t n = 0);
void rope(Tensor& xout, const Tensor& x, const Tensor& cos, const Tensor& sin);
void silu(Tensor& xout, const Tensor& x);
void add(Tensor& xout, const Tensor& x, const Tensor& y);
void mul(Tensor& xout, const Tensor& x, const Tensor& y);
void mul(Tensor& xout, const Tensor& x, float c);
