#include "backend/metal/kernels.h"

void matmul(Tensor& xout, const Tensor& w, const Tensor& x, size_t m) {
    (void)xout;
    (void)w;
    (void)x;
    (void)m;
}

void row_matmul(Tensor& xout, const Tensor& x, const Tensor& w, size_t n) {
    (void)xout;
    (void)x;
    (void)w;
    (void)n;
}

void softmax(Tensor& xout, const Tensor& x, size_t n) {
    (void)xout;
    (void)x;
    (void)n;
}

void rope(Tensor& xout, const Tensor& x, const Tensor& cos, const Tensor& sin) {
    (void)xout;
    (void)x;
    (void)cos;
    (void)sin;
}

void silu(Tensor& xout, const Tensor& x) {
    (void)xout;
    (void)x;
}

void add(Tensor& xout, const Tensor& x, const Tensor& y) {
    (void)xout;
    (void)x;
    (void)y;
}

void mul(Tensor& xout, const Tensor& x, const Tensor& y) {
    (void)xout;
    (void)x;
    (void)y;
}

void mul(Tensor& xout, const Tensor& x, float c) {
    (void)xout;
    (void)x;
    (void)c;
}
