#include <algorithm>
#include <cmath>
#include <math.h>
#include <vector>
#include "kernels.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

template <typename WeightT, typename ActivationT, typename AccumT>
AccumT dot_product(Tensor<WeightT>& w, size_t row, Tensor<ActivationT>& x, size_t d);

template <>
float dot_product<float, float, float>(Tensor<float>& w, size_t row, Tensor<float>& x, size_t d) {
    const float* a = w.data + row * d;
    const float* b = x.data;
#if defined(__ARM_NEON)
    float32x4_t sum = vdupq_n_f32(0.f);
    size_t i = 0;
    for (; i + 4 <= d; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vfmaq_f32(sum, va, vb);
    }
    float result = vaddvq_f32(sum);
    for (; i < d; i++) {
        result += a[i] * b[i];
    }
    return result;
#else
    float result = 0.f;
    for (size_t i = 0; i < d; i++) {
        result += a[i] * b[i];
    }
    return result;
#endif
}

template <>
float dot_product<fp16_t, float, float>(Tensor<fp16_t>& w, size_t row, Tensor<float>& x, size_t d) {
    const fp16_t* row_data = w.data + row * d;
#if defined(__ARM_NEON)
    float16x8_t acc = vdupq_n_f16(0);
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        float16x8_t wh = vld1q_f16(reinterpret_cast<const __fp16*>(row_data + i));
        float16x4_t xf_lo = vcvt_f16_f32(vld1q_f32(x.data + i));
        float16x4_t xf_hi = vcvt_f16_f32(vld1q_f32(x.data + i + 4));
        float16x8_t xf = vcombine_f16(xf_lo, xf_hi);
        acc = vfmaq_f16(acc, wh, xf);
    }

    fp16_t lanes[8];
    vst1q_f16(reinterpret_cast<__fp16*>(lanes), acc);

    fp16_t result_h = f32_to_fp16(0.f);
    for (size_t j = 0; j < 8; j++) {
        result_h = f32_to_fp16(fp16_to_f32(result_h) + fp16_to_f32(lanes[j]));
    }
    for (; i < d; i++) {
        result_h = f32_to_fp16(
            fp16_to_f32(result_h) + fp16_to_f32(row_data[i]) * x.data[i]
        );
    }
    return fp16_to_f32(result_h);
#else
    fp16_t result_h = f32_to_fp16(0.f);
    for (size_t i = 0; i < d; i++) {
        result_h = f32_to_fp16(
            fp16_to_f32(result_h) + fp16_to_f32(row_data[i]) * x.data[i]
        );
    }
    return fp16_to_f32(result_h);
#endif
}

template <>
float dot_product<int8_t, float, float>(Tensor<int8_t>& w, size_t row, Tensor<float>& x, size_t d) {
    constexpr size_t GROUP_SIZE = 64;
    const size_t num_groups = d / GROUP_SIZE;
    const size_t nscales = w.scales.size();
    const size_t numel = w.numel;
    const size_t w_offset = row * d;

    float sum = 0.0f;

    for (size_t g=0; g<num_groups; g++){
        size_t group_start = g * GROUP_SIZE;
        size_t scale_idx = (nscales * (w_offset + group_start)) / numel;
        float inv_scale = 1.0f / w.scales[scale_idx];
        const int8_t* group_w = w.data + w_offset + group_start;
        const float* group_x = x.data + group_start;

#if defined(__ARM_NEON)
        float32x4_t acc = vdupq_n_f32(0.f);
        size_t j = 0;
        for (; j + 8 <= GROUP_SIZE; j += 8) {
            int8x8_t wi = vld1_s8(group_w + j);
            int16x8_t w16 = vmovl_s8(wi);
            float32x4_t wf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
            float32x4_t wf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));
            float32x4_t xf_lo = vld1q_f32(group_x + j);
            float32x4_t xf_hi = vld1q_f32(group_x + j + 4);
            acc = vfmaq_f32(acc, wf_lo, xf_lo);
            acc = vfmaq_f32(acc, wf_hi, xf_hi);
        }
        float group_sum = vaddvq_f32(acc);
        for (; j < GROUP_SIZE; j++) {
            group_sum += group_w[j] * group_x[j];
        }
#else
        float group_sum = 0.f;
        for (size_t j = 0; j < GROUP_SIZE; j++) {
            group_sum += group_w[j] * group_x[j];
        }
#endif

        sum += inv_scale * group_sum;
    }

    size_t remaining_start = num_groups * GROUP_SIZE;
    for (size_t j = remaining_start; j < d; j++){
        sum += w.get(w_offset + j) * x.data[j];
    }

    return sum;
}

template <typename WeightT, typename ActivationT, typename AccumT>
void matmul(Tensor<AccumT>& xout, Tensor<WeightT>& w, Tensor<ActivationT>& x) {
    size_t n = w.shape[0];
    size_t d = w.shape[1];

    assert(x.numel == d && xout.numel >= n && "matmul shape mismatch");

    #pragma omp parallel for
    for (int i=0; i<(int)n; i++){
        xout.data[i] = dot_product<WeightT, ActivationT, AccumT>(w, static_cast<size_t>(i), x, d);
    }
}

template <>
void matmul<int8_t, float, float>(Tensor<float>& xout, Tensor<int8_t>& w, Tensor<float>& x) {
    size_t n = w.shape[0];
    size_t d = w.shape[1];

    assert(x.numel == d && xout.numel >= n && "matmul shape mismatch");

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    constexpr size_t GROUP_SIZE = 64;
    const size_t num_groups = d / GROUP_SIZE;
    const size_t nscales = w.scales.size();
    const size_t numel = w.numel;

    std::vector<int8_t> xq(d);
    std::vector<float> x_scales(num_groups, 1.f);

    for (size_t g = 0; g < num_groups; g++) {
        const size_t group_start = g * GROUP_SIZE;
        float max_abs = 0.f;
        for (size_t j = 0; j < GROUP_SIZE; j++) {
            max_abs = std::max(max_abs, std::fabs(x.data[group_start + j]));
        }

        if (max_abs == 0.f) {
            continue;
        }

        const float x_scale = 127.f / max_abs;
        x_scales[g] = x_scale;
        for (size_t j = 0; j < GROUP_SIZE; j++) {
            int q = static_cast<int>(std::round(x.data[group_start + j] * x_scale));
            q = std::max(-127, std::min(127, q));
            xq[group_start + j] = static_cast<int8_t>(q);
        }
    }

    #pragma omp parallel for
    for (int row = 0; row < static_cast<int>(n); row++) {
        const size_t w_offset = static_cast<size_t>(row) * d;
        float sum = 0.f;

        for (size_t g = 0; g < num_groups; g++) {
            const size_t group_start = g * GROUP_SIZE;
            const size_t scale_idx = (nscales * (w_offset + group_start)) / numel;
            const float dequant_scale = 1.f / (w.scales[scale_idx] * x_scales[g]);
            const int8_t* group_w = w.data + w_offset + group_start;
            const int8_t* group_x = xq.data() + group_start;

            int32x4_t acc = vdupq_n_s32(0);
            size_t j = 0;
            for (; j + 16 <= GROUP_SIZE; j += 16) {
                int8x16_t wi = vld1q_s8(group_w + j);
                int8x16_t xi = vld1q_s8(group_x + j);
                acc = vdotq_s32(acc, wi, xi);
            }

            int32_t group_sum = vaddvq_s32(acc);
            for (; j < GROUP_SIZE; j++) {
                group_sum += static_cast<int32_t>(group_w[j]) * static_cast<int32_t>(group_x[j]);
            }

            sum += dequant_scale * static_cast<float>(group_sum);
        }

        const size_t remaining_start = num_groups * GROUP_SIZE;
        for (size_t j = remaining_start; j < d; j++) {
            sum += w.get(w_offset + j) * x.data[j];
        }

        xout.data[row] = sum;
    }
#else
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(n); i++) {
        xout.data[i] = dot_product<int8_t, float, float>(w, static_cast<size_t>(i), x, d);
    }
#endif
}

template void matmul<float, float, float>(Tensor<float>&, Tensor<float>&, Tensor<float>&);
template void matmul<fp16_t, float, float>(Tensor<float>&, Tensor<fp16_t>&, Tensor<float>&);


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

void row_matmul(Tensor<float>& xout, Tensor<float>& x, Tensor<fp16_t>& w){
    size_t n = w.shape[0];
    size_t d = w.shape[1];

    assert(x.shape[0] == n && xout.numel >= d && "matmul shape mismatch");

    #pragma omp parallel for
    for (int i=0; i<(int)d; i++){
        float sum = 0.f;
        for (size_t j=0; j<n; j++){
            sum += fp16_to_f32(w.data[j * d + i]) * x.data[j];
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

void add(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& y){
    for (int i = 0; i < x.numel; i++) {
        xout.data[i] = x.data[i] + y.data[i];
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

