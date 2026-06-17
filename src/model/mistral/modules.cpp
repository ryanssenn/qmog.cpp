#include "../../../include/modules.h"
#include <iostream>

// Assumes only 1 id
void Embedding::forward(InferenceState& infer, size_t token_id){
    infer.hidden_state.copy_from(table.at({token_id}));
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
void RMSNorm::forward(InferenceState& infer) {
    float squares = 0;

    for(int i =0; i<infer.hidden_state.numel; i++){
        squares += infer.hidden_state.data[i] * infer.hidden_state.data[i];
    }

    float rms = sqrt(squares/infer.hidden_state.shape[0] + e);

    mul(infer.hidden_state, infer.hidden_state,1/rms);

    // Element wise mul with g
    for (int i=0; i<infer.hidden_state.numel; i++){
        infer.hidden_state.data[i] = infer.hidden_state.data[i] * g.data[i];
    }
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L140
void Attention::forward(InferenceState &infer) {
    // Get q, k, v
    // [proj, hidden_size] @ [hidden_size, 1] = [proj]
    matmul(infer.q_state, q_proj, infer.hidden_state);
    matmul(infer.k_state, k_proj, infer.hidden_state);
    matmul(infer.v_state, v_proj, infer.hidden_state);

    // Populate cos/sin embeddings
    RotaryEmbedding::forward(infer);

    // Rotate Q,K
    rope(infer.q_state, infer.q_state, infer.cos, infer.sin);
    rope(infer.k_state, infer.k_state, infer.cos, infer.sin);

    // Push KV to cache
    infer.push_kv(layer);

    // Perform attention with tokens in window
    // softmax ( QK^t / sqrt(head_dim) ) * V
    // Reuse each KV head 4 times
    #pragma omp parallel for
    for (size_t h=0; h<infer.config.n_heads; h++){
        // [seq_len, 128] @ [128]
        Tensor q_head = infer.q_state.at({h}); // [128]
        Tensor k_head = infer.k_cache.at({layer, h/4}).reshape({infer.pos+1, infer.config.head_dim}); // [seq_len, 128]
        Tensor score_head = infer.scores.at({h}).reshape({infer.pos+1});

        // KQ
        matmul(score_head, k_head, q_head);
        // Divide by dk
        mul(score_head, score_head, 1/sqrt(infer.config.head_dim));
        // Softmax
        softmax(score_head, score_head); // [seq_len]

        Tensor v_head = infer.v_cache.at({layer, h/4}).reshape({infer.pos+1, infer.config.head_dim});  // [seq_len, 128]
        Tensor context_head = infer.context.at({h});

        // score_head [seq_len] @ v_head [seq_len, head_dim]
        row_matmul(context_head, score_head, v_head);
    }

    // o_proj [4096, 4096] @ context [4096]
    matmul(infer.hidden_state, o_proj, infer.context);
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L46
// Mistral uses a SwiGLU feedforward block
// It runs the input through two linear projections.
// The first gives the main signal, the second goes through a silu activation
// Then multiplies these two paths together and apply the final projection
template <typename TGateUp>
void MLP<TGateUp>::forward(InferenceState &infer) {
    // gate_proj [14336, 4096] @ hidden_state [4096]
    matmul(infer.mlp_gate, gate_proj, infer.hidden_state);

    // Activation
    silu(infer.mlp_gate, infer.mlp_gate);

    // up_proj [14336, 4096] @ hidden_state [4096]
    matmul(infer.mlp_up, up_proj, infer.hidden_state);

    // Multiply
    mul(infer.mlp_gate, infer.mlp_gate, infer.mlp_up);

    // down_proj [4096, 14336] @ [14336]
    matmul(infer.hidden_state, down_proj, infer.mlp_gate);
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L215
template <typename TMlp>
void Layer<TMlp>::forward(InferenceState &infer){
    // Copy input into residual
    infer.residual.copy_from(infer.hidden_state);

    // Layer norm
    input_norm.forward(infer);

    // Self attention
    attn.forward(infer);

    // Add residual to hidden state
    add(infer.hidden_state, infer.hidden_state, infer.residual);

    // Copy input into residual
    infer.residual.copy_from(infer.hidden_state);

    // Layer norm
    output_norm.forward(infer);

    // Feed forward
    mlp.forward(infer);

    // Add residual
    add(infer.hidden_state, infer.hidden_state, infer.residual);
}


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L430
void LMHead::forward(InferenceState &infer) {
    matmul(infer.logits, lm_head, infer.hidden_state);
}


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L334
// Processes 1 token at a time and do a rewrite to support multiple tokens
template <typename TMlp>
void Model<TMlp>::forward(InferenceState &infer, size_t token_id) {
    // Get token embedding
    embedding.forward(infer, token_id);

    // Forward each layer
    for (auto& layer : layers) {
        layer.forward(infer);
    }

    // Norm
    norm.forward(infer);

    // Get logits
    lmHead.forward(infer);

    infer.pos++;
}


template void MLP<float>::forward(InferenceState &);
template void Layer<float>::forward(InferenceState &);
template void Model<float>::forward(InferenceState &, size_t);

template void MLP<int8_t>::forward(InferenceState &);
template void Layer<int8_t>::forward(InferenceState &);
template void Model<int8_t>::forward(InferenceState &, size_t);