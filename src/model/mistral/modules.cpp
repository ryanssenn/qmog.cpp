#include "modules.h"
#include <iostream>

template <typename TLinear>
void Embedding<TLinear>::forward(InferenceState& infer, size_t token_id){
    Tensor<TLinear> row = table.at({token_id});
    for (size_t i = 0; i < embedding_dim; i++) {
        infer.hidden_state.data[i] = row.get(i);
    }
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L290
// Computes RoPE inverse frequencies
// inv_freq[i] = (1 / rope_theta^(i / (head_dim))) / factor
void RotaryEmbedding::init_freq(InferenceState& infer, Config& config) {
    for (int i=0;i<infer.inv_freq.numel;i++){
        float freq = 1.0f / std::pow(config.rope_theta, float(i)/infer.inv_freq.numel);
        infer.inv_freq.data[i] = freq;
    }
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L319
// The forward pass generates cos/sin position encodings for RoPE.
// Take the inv_freq at each position, multiply them by position and apply cos/sin
void RotaryEmbedding::forward(InferenceState& infer){
    for (size_t i=0; i<infer.cos.numel; i++){
        infer.cos.data[i] = std::cos(infer.inv_freq.data[i % infer.inv_freq.numel] * infer.pos);
        infer.sin.data[i] = std::sin(infer.inv_freq.data[i % infer.inv_freq.numel] * infer.pos);
    }
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L195
// x*g / sqrt(sum(x^2) + e)
template <typename TLinear>
void RMSNorm<TLinear>::forward(InferenceState& infer) {
    float squares = 0;

    for(int i =0; i<infer.hidden_state.numel; i++){
        squares += infer.hidden_state.data[i] * infer.hidden_state.data[i];
    }

    float rms = sqrt(squares/infer.hidden_state.shape[0] + e);

    mul(infer.hidden_state, infer.hidden_state,1/rms);

    for (int i=0; i<infer.hidden_state.numel; i++){
        infer.hidden_state.data[i] = infer.hidden_state.data[i] * g.get(i);
    }
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L140
template <typename TLinear>
void Attention<TLinear>::forward(InferenceState &infer) {
    matmul(infer.q_state, q_proj, infer.hidden_state);
    matmul(infer.k_state, k_proj, infer.hidden_state);
    matmul(infer.v_state, v_proj, infer.hidden_state);

    RotaryEmbedding::forward(infer);

    rope(infer.q_state, infer.q_state, infer.cos, infer.sin);
    rope(infer.k_state, infer.k_state, infer.cos, infer.sin);

    infer.push_kv(layer);

    #pragma omp parallel for
    for (size_t h=0; h<infer.config.n_heads; h++){
        auto q_head = infer.q_state.at({h});
        auto k_head = infer.k_cache.at({layer, h/4}).reshape({infer.pos+1, infer.config.head_dim});
        auto score_head = infer.scores.at({h}).reshape({infer.pos+1});

        matmul(score_head, k_head, q_head);
        mul(score_head, score_head, 1/sqrt(infer.config.head_dim));
        softmax(score_head, score_head);

        auto v_head = infer.v_cache.at({layer, h/4}).reshape({infer.pos+1, infer.config.head_dim});
        auto context_head = infer.context.at({h});

        row_matmul(context_head, score_head, v_head);
    }

    matmul(infer.hidden_state, o_proj, infer.context);
}

template <typename TGateUp, typename TLinear>
void MLP<TGateUp, TLinear>::forward(InferenceState &infer) {
    matmul(infer.mlp_gate, gate_proj, infer.hidden_state);

    silu(infer.mlp_gate, infer.mlp_gate);

    matmul(infer.mlp_up, up_proj, infer.hidden_state);

    mul(infer.mlp_gate, infer.mlp_gate, infer.mlp_up);

    matmul(infer.hidden_state, down_proj, infer.mlp_gate);
}

template <typename TGateUp, typename TLinear>
void Layer<TGateUp, TLinear>::forward(InferenceState &infer){
    infer.residual.copy_from(infer.hidden_state);

    input_norm.forward(infer);

    attn.forward(infer);

    add(infer.hidden_state, infer.hidden_state, infer.residual);

    infer.residual.copy_from(infer.hidden_state);

    output_norm.forward(infer);

    mlp.forward(infer);

    add(infer.hidden_state, infer.hidden_state, infer.residual);
}

template <typename TLinear>
void LMHead<TLinear>::forward(InferenceState &infer) {
    matmul(infer.logits, lm_head, infer.hidden_state);
}

template <typename TGateUp, typename TLinear>
void Model<TGateUp, TLinear>::forward(InferenceState &infer, size_t token_id) {
    embedding.forward(infer, token_id);

    for (auto& layer : layers) {
        layer.forward(infer);
    }

    norm.forward(infer);

    lmHead.forward(infer);

    infer.pos++;
}

template void MLP<float, float>::forward(InferenceState &);
template void Layer<float, float>::forward(InferenceState &);
template void Model<float, float>::forward(InferenceState &, size_t);
template void Embedding<float>::forward(InferenceState&, size_t);
template void RMSNorm<float>::forward(InferenceState&);
template void LMHead<float>::forward(InferenceState&);
template void Attention<float>::forward(InferenceState&);

template void MLP<int8_t, float>::forward(InferenceState &);
template void Layer<int8_t, float>::forward(InferenceState &);
template void Model<int8_t, float>::forward(InferenceState &, size_t);

template void MLP<int8_t, fp16_t>::forward(InferenceState &);
template void Layer<int8_t, fp16_t>::forward(InferenceState &);
template void Model<int8_t, fp16_t>::forward(InferenceState &, size_t);
template void Embedding<fp16_t>::forward(InferenceState&, size_t);
template void RMSNorm<fp16_t>::forward(InferenceState&);
template void LMHead<fp16_t>::forward(InferenceState&);
template void Attention<fp16_t>::forward(InferenceState&);
