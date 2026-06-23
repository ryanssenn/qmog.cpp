#pragma once
#include "tensor.h"
#include "parameters.h"
#include "fp16.h"

inline size_t MAX_SEQ_LEN = 500;

// Holds Tensor<float> memory used during inference
struct InferenceState {
    Config config;
    Arena arena;

    Tensor<float> hidden_state; // [hidden_size]
    Tensor<float> residual; // [hidden_size]
    size_t pos = 0;

    Tensor<float> inv_freq; // [head_dim / 2]
    Tensor<float> cos; // [head_dim]
    Tensor<float> sin; // [head_dim]

    Tensor<float> q_state; // [n_heads, head_dim]
    Tensor<float> k_state; // [n_kv_heads, head_dim]
    Tensor<float> v_state; // [n_kv_heads, head_dim]

    Tensor<fp16_t> k_cache; // [n_layer, n_kv_heads, seq_len, head_dim]
    Tensor<fp16_t> v_cache; // [n_layer, n_kv_heads, seq_len, head_dim]

    Tensor<float> scores; // [n_heads, seq_len]
    Tensor<float> context; // [n_heads, head_dim]

    Tensor<float> mlp_gate; // [intermediate_size]
    Tensor<float> mlp_up; // [intermediate_size]

    Tensor<float> logits; // [vocab_size]

    Tensor<float> probs; // [vocab_size]

    void push_kv(size_t i){
        for (size_t h=0;h<config.n_kv_heads;h++){
            Tensor<fp16_t> k_dst = k_cache.at({i, h, pos});
            Tensor<fp16_t> v_dst = v_cache.at({i, h, pos});
            Tensor<float> k_src = k_state.at({h});
            Tensor<float> v_src = v_state.at({h});

            for (size_t j = 0; j < config.head_dim; j++) {
                k_dst.data[j] = f32_to_fp16(k_src.data[j]);
                v_dst.data[j] = f32_to_fp16(v_src.data[j]);
            }
        }
    }

    InferenceState(Config& config) : config(config),
                                     arena(256 * 1024 * 1024), // 400MB, how much memory will be needed?
                                     hidden_state(arena, {config.hidden_size}), // Only 1 token at a time, pretty sure i will be having to rewrite this
                                     residual(arena, {config.hidden_size}),

                                     inv_freq(arena, {config.head_dim / 2}),
                                     cos(arena, {config.head_dim}),
                                     sin(arena, {config.head_dim}),

                                     q_state(arena, {config.n_heads, config.head_dim}),
                                     k_state(arena, {config.n_kv_heads, config.head_dim}),
                                     v_state(arena, {config.n_kv_heads, config.head_dim}),

                                     k_cache(arena, {config.n_layers, config.n_kv_heads, MAX_SEQ_LEN, config.head_dim}),
                                     v_cache(arena, {config.n_layers, config.n_kv_heads, MAX_SEQ_LEN, config.head_dim}),

                                     scores(arena, {config.n_heads, MAX_SEQ_LEN}),
                                     context(arena, {config.n_heads, config.head_dim}),

                                     mlp_gate(arena, {config.intermediate_size}),
                                     mlp_up(arena, {config.intermediate_size}),

                                     logits(arena, {config.vocab_size}),

                                     probs(arena, {config.vocab_size})

                                     {}
};