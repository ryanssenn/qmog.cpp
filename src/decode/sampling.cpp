#include "decode/sampling.h"

#include "common/tensor.h"

#include <algorithm>
#include <cmath>
#include <random>

static void softmax_cpu(Tensor& xout, const Tensor& x) {
    float* xout_data = xout.f32();
    const float* x_data = x.f32();
    float max_val = x_data[0];
    for (size_t i = 1; i < x.numel; i++) {
        max_val = std::max(max_val, x_data[i]);
    }

    float total = 0.f;
    for (size_t i = 0; i < x.numel; i++) {
        xout_data[i] = std::exp(x_data[i] - max_val);
        total += xout_data[i];
    }

    for (size_t i = 0; i < x.numel; i++) {
        xout_data[i] /= total;
    }
}

// Argmax over the vocabulary — greedy decoding when temperature is 0.
uint32_t sample_max(const InferenceState& infer) {
    const float* logits = infer.logits.f32();
    uint32_t best = 0;
    float best_logit = logits[0];
    for (size_t i = 1; i < infer.config.vocab_size; i++) {
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best = static_cast<uint32_t>(i);
        }
    }
    return best;
}

// Sample from softmax(logits / temp). Falls back to greedy when temp <= 0.
uint32_t sample_multinomial(InferenceState& infer, float temp) {
    if (temp <= 0.0f) {
        return sample_max(infer);
    }

    float* probs = infer.probs.f32();
    const float* logits = infer.logits.f32();
    for (size_t i = 0; i < infer.logits.numel; i++) {
        probs[i] = logits[i] / temp;
    }
    softmax_cpu(infer.probs, infer.probs);

    static std::mt19937 gen{std::random_device{}()};
    static std::uniform_real_distribution<double> uniform(0.0, 1.0);

    const double threshold = uniform(gen);
    double cumulative = 0.0;
    for (size_t i = 0; i < infer.probs.numel; i++) {
        cumulative += probs[i];
        if (cumulative >= threshold) {
            return static_cast<uint32_t>(i);
        }
    }
    return static_cast<uint32_t>(infer.probs.numel - 1);
}
