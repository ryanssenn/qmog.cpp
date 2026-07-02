#include "model/model.h"

#include "backend/metal/kernels.h"

#include <cassert>


void Embedding::forward(InferenceState& infer, size_t token_id){
    Tensor row = table.at({token_id});
    for (size_t i = 0; i < embedding_dim; i++) {
        infer.hidden_state.f32()[i] = static_cast<float>(row.f16()[i]);
    }
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py#L106
void RotaryEmbedding::forward(InferenceState& infer){
    for (size_t i=0; i<infer.cos.numel; i++){
        infer.cos.f32()[i] = std::cos(infer.inv_freq.f32()[i % infer.inv_freq.numel] * infer.pos);
        infer.sin.f32()[i] = std::sin(infer.inv_freq.f32()[i % infer.inv_freq.numel] * infer.pos);
    }
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py#L58
void RMSNorm::forward(Tensor& x) {
    const size_t cols = x.shape[x.ndim - 1];
    assert(cols == g.numel);

    float* data = x.f32();
    for (size_t s = 0; s < x.numel / cols; s++) {
        float* slice = data + s * cols;

        float squares = 0;
        for (size_t i = 0; i < cols; i++) {
            squares += slice[i] * slice[i];
        }

        const float scale = 1.0f / sqrt(squares / cols + e);
        for (size_t i = 0; i < cols; i++) {
            slice[i] *= scale * static_cast<float>(g.f16()[i]);
        }
    }
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py#L222
void Attention::forward(InferenceState &infer) {
    matmul(infer.q_state, q_proj, infer.hidden_state);
    matmul(infer.k_state, k_proj, infer.hidden_state);
    q_norm.forward(infer.q_state);
    k_norm.forward(infer.k_state);
    matmul(infer.v_state, v_proj, infer.hidden_state);

    RotaryEmbedding::forward(infer);

    rope(infer.q_state, infer.q_state, infer.cos, infer.sin);
    rope(infer.k_state, infer.k_state, infer.cos, infer.sin);

    infer.push_kv(layer);

    for (size_t h=0; h<infer.config.n_heads; h++){
        auto q_head = infer.q_state.at({h});
        auto k_head = infer.k_cache.at({layer, h/4});
        auto score_head = infer.scores.at({h});

        matmul(score_head, k_head, q_head, infer.pos + 1);
        mul(score_head, score_head, 1/sqrt(infer.config.head_dim));
        softmax(score_head, score_head, infer.pos + 1);

        auto v_head = infer.v_cache.at({layer, h/4});
        auto context_head = infer.context.at({h});

        row_matmul(context_head, score_head, v_head, infer.pos + 1);
    }

    matmul(infer.hidden_state, o_proj, infer.context);
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py#L222
void MLP::forward(InferenceState &infer) {
    matmul(infer.mlp_gate, gate_proj, infer.hidden_state);

    silu(infer.mlp_gate, infer.mlp_gate);

    matmul(infer.mlp_up, up_proj, infer.hidden_state);

    mul(infer.mlp_gate, infer.mlp_gate, infer.mlp_up);

    matmul(infer.hidden_state, down_proj, infer.mlp_gate);
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py#L222
void Layer::forward(InferenceState &infer){
    infer.residual.copy_from(infer.hidden_state);

    input_norm.forward(infer.hidden_state);

    attn.forward(infer);

    add(infer.hidden_state, infer.hidden_state, infer.residual);

    infer.residual.copy_from(infer.hidden_state);

    output_norm.forward(infer.hidden_state);

    mlp.forward(infer);

    add(infer.hidden_state, infer.hidden_state, infer.residual);
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py#L222
void LMHead::forward(InferenceState &infer) {
    matmul(infer.logits, lm_head, infer.hidden_state);
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py#L222
void Model::forward(InferenceState &infer, size_t token_id) {
    embedding.forward(infer, token_id);

    for (auto& layer : layers) {
        layer.forward(infer);
    }

    norm.forward(infer.hidden_state);

    lmHead.forward(infer);

    infer.pos++;
}
