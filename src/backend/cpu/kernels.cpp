#include <math.h>
#include "../../../include/kernels.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static float dot_f32(const float* a, const float* b, size_t n) {
#if defined(__ARM_NEON)
    float32x4_t sum = vdupq_n_f32(0.f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vfmaq_f32(sum, va, vb);
    }
    float result = vaddvq_f32(sum);
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float result = _mm_cvtss_f32(sums);
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
#else
    float result = 0.f;
    for (size_t i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
#endif
}

static float dot_i8_f32(const int8_t* w, const float* x, size_t n) {
#if defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int8x8_t wi = vld1_s8(w + i);
        int16x8_t w16 = vmovl_s8(wi);
        float32x4_t wf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
        float32x4_t wf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));
        float32x4_t xf_lo = vld1q_f32(x + i);
        float32x4_t xf_hi = vld1q_f32(x + i + 4);
        acc = vfmaq_f32(acc, wf_lo, xf_lo);
        acc = vfmaq_f32(acc, wf_hi, xf_hi);
    }
    float result = vaddvq_f32(acc);
    for (; i < n; i++) {
        result += w[i] * x[i];
    }
    return result;
#else
    float result = 0.f;
    for (size_t i = 0; i < n; i++) {
        result += w[i] * x[i];
    }
    return result;
#endif
}


// W (n,d) @ x (d,) = xout (n,)
void matmul(Tensor<float>& xout, Tensor<float>& w, Tensor<float>& x){
    size_t n = w.shape[0];
    size_t d = w.shape[1];

    assert(x.numel == d && xout.numel >= n && "matmul shape mismatch");

    #pragma omp parallel for
    for (int i=0; i<(int)n; i++){
        xout.data[i] = dot_f32(w.data + i * d, x.data, d);
    }
}

void matmul(Tensor<float>& xout, Tensor<int8_t>& w, Tensor<float>& x){
    size_t n = w.shape[0];
    size_t d = w.shape[1];

    assert(x.numel == d && xout.numel >= n && "matmul shape mismatch");

    const size_t nscales = w.scales.size();
    const size_t numel = w.numel;
    assert(nscales > 0 && numel % nscales == 0 && "invalid int8 scale metadata");
    const size_t group_size = numel / nscales;
    assert(d % group_size == 0 && "int8 group size must align with matmul rows");
    const size_t num_groups = d / group_size;

    #pragma omp parallel for
    for (int i=0; i<(int)n; i++){
        float sum = 0.0f;
        size_t w_offset = i * d;

        for (size_t g=0; g<num_groups; g++){
            size_t group_start = g * group_size;
            size_t scale_idx = (w_offset + group_start) / group_size;
            float inv_scale = 1.0f / w.scales[scale_idx];
            sum += inv_scale * dot_i8_f32(
                w.data + w_offset + group_start,
                x.data + group_start,
                group_size
            );
        }

        size_t remaining_start = num_groups * group_size;
        for (size_t j = remaining_start; j < d; j++){
            sum += w.get(w_offset + j) * x.data[j];
        }

        xout.data[i] = sum;
    }
}


// x (,n) @ W (n,d) = xout (d,)
void row_matmul(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& w){
    size_t n = w.shape[0];
    size_t d = w.shape[1];

    assert(x.shape[0] == n && xout.numel >= d && "matmul shape mismatch");

    #pragma omp parallel for
    for (int i=0; i<(int)d; i++){
        float sum = 0.f;
        for (size_t j=0; j<n; j++){
            sum += w.data[j * d + i] * x.data[j];
        }
        xout.data[i] = sum;
    }
}


// e^x / sum(e^x)
// Here we compute the max and subtract it from each logit to avoid overflow,
// keeping the relative ratios unchanged (softmax is shift-invariant)
void softmax(Tensor<float>& xout, Tensor<float>& x){
    float maxv = x.max();
    float total = 0;
    for (int i=0; i<x.numel; i++){
        xout.data[i] = std::exp(x.data[i] - maxv);
        total += xout.data[i];
    }
    mul(xout, xout, 1/total);
}


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L58
// We perform 2D rotations each pair
// [x', y'] = [x*cosθ - y*sinθ, x*sinθ + y*cosθ]
// The same cos/sin embedding is reused for every head
// Inputs:
//   x: [n_heads, seq_len, head_dim] (q or k)
//   cos, sin: [seq_len, head_dim]

// Original RoPE rotates consecutive dim pairs (i, i+1), but Mistral uses a half-split layout?
// Each dim i is rotated with i + head_dim/2

// TODO: This only works 1 token at time (assumed seq_len = 1), cos/sin is only given for 1 pos
void rope(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& cos, Tensor<float>& sin){
    size_t n_heads = x.shape[0];
    //size_t seq_len = x.shape[1];
    size_t head_size = x.shape[1];
    size_t half = head_size/2;

    for (size_t h = 0; h<n_heads; h++){
        int start = h*x.strides[0];
        for (int i = start; i < start+half; i++){
            float xi = x.data[i];
            float yi = x.data[i+half];
            float c = cos.data[i - start];
            float s = sin.data[i - start];
            xout.data[i] = xi*c - yi*s;
            xout.data[i+half] = xi*s + yi*c;
        }
    }
}

// x / (1 + exp(-x))
void silu(Tensor<float>& xout, Tensor<float>& x){
    for (int i=0;i<x.numel;i++){
        xout.data[i] = x.data[i] / (1 + exp(-x.data[i]));
    }
}


// Not sure if I will be using all of those
float sum(Tensor<float>& x){
    float r = 0.0f;
    for (int i=0; i<x.numel; i++){
        r += x.data[i];
    }
    return r;
}

void add(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& y){
    for (int i = 0; i < x.numel; i++) {
        xout.data[i] = x.data[i] + y.data[i];
    }
}


void add(Tensor<float>& xout, Tensor<float>& x, float c){
    for (int i = 0; i < x.numel; i++) {
        xout.data[i] = x.data[i] + c;
    }
}

void mul(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& y){
    for (int i = 0; i < x.numel; i++) {
        xout.data[i] = x.data[i] * y.data[i];
    }
}

void mul(Tensor<float>& xout, Tensor<float>& x, float c) {
    for (int i = 0; i < x.numel; i++) {
        xout.data[i] = x.data[i] * c;
    }
}

void pow(Tensor<float>& xout, Tensor<float>& x, int e){
    for (int i=0; i<x.numel; i++){
        xout.data[i] = pow(x.data[i], e);
    }
}

void sqrt(Tensor<float>& xout, Tensor<float>& x){
    for (int i=0; i<x.numel; i++){
        xout.data[i] = sqrt(x.data[i]);
    }
}
