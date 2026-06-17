#include "setup/context.h"

template <typename T>
int test_layer() {
    std::shared_ptr<Parameters> params = get_params();
    infer.pos = 0;

    Layer<T> layer(0, params);

    // Test over sequence of 3 tokens
    for (int i=1;i<4;i++) {
        infer.hidden_state.copy_from(expected.at("layer_h" + std::to_string(i)));
        layer.forward(infer);

        if (!equals(infer.hidden_state, expected.at("layer_o" + std::to_string(i)))){
            std::cout << "Layer mismatch at token " + std::to_string(i) << std::endl;
            return 1;
        }
    }

    return 0;
}

RegisterTest layer_reg("test layer", "f32", &test_layer<float>);

template <typename T>
int test_attention() {
    std::shared_ptr<Parameters> params = get_params();
    infer.pos = 0;

    auto& w = params->layer_weights[0];
    Attention attn(params->get_tensor<T>(0, "self_attn.q_proj.weight"), params->get_tensor<T>(0, "self_attn.k_proj.weight"), params->get_tensor<T>(0, "self_attn.v_proj.weight"), params->get_tensor<T>(0, "self_attn.o_proj.weight"), 0);

    // Test over sequence of 3 tokens
    for (int i=1;i<4;i++){
        infer.hidden_state.copy_from(expected.at("attn_h" + std::to_string(i)));
        attn.forward(infer);

        if (!equals(infer.q_state, expected.at("attn_q" + std::to_string(i)))){
            std::cout << "Attention q mismatch at token " + std::to_string(i) << std::endl;
            return 1;
        }

        if (!equals(infer.k_state, expected.at("attn_k" + std::to_string(i)))){
            std::cout << "Attention k mismatch at token "  + std::to_string(i)  << std::endl;
            return 1;
        }

        if (!equals(infer.v_state, expected.at("attn_v" + std::to_string(i)))){
            std::cout << "Attention v mismatch at token "  + std::to_string(i)  << std::endl;
            return 1;
        }

        /*
        if (!equals(infer.scores.reshape({infer.scores.shape[0], infer.pos+1}), expected.at("attn_w" + std::to_string(i)))){
            std::cout << "Attention weights mismatch at token " + std::to_string(i)  << std::endl;
            return 1;
        }
        */

        if (!equals(infer.hidden_state, expected.at("attn_o" + std::to_string(i)))){
            std::cout << "Attention result mismatch at token " + std::to_string(i)  << std::endl;
            return 1;
        }
        infer.pos++;
    }

    return 0;
}

RegisterTest attention_reg("test attention", "f32", &test_attention<float>);
//RegisterTest attention_reg_q("test attention", "int8", &test_attention<int8_t>);


template <typename T>
int test_mlp(){
    std::shared_ptr<Parameters> params = get_params();

    MLP<T> mlp(params->get_tensor<float>(0, "mlp.down_proj.weight"),
               params->get_tensor<T>(0, "mlp.gate_proj.weight"),
               params->get_tensor<T>(0, "mlp.up_proj.weight"));

    infer.hidden_state.copy_from(expected.at("mlp_h"));

    mlp.forward(infer);

    if (!equals(infer.hidden_state, expected.at("mlp_output"))){
        std::cout << "MLP feedforward result mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest mlp_feedforward_reg("test attention feedforward mlp", "f32", &test_mlp<float>);
RegisterTest mlp_feedforward_reg_q("test attention feedforward mlp", "int8", &test_mlp<int8_t>);

int test_kv_cache() {
    infer.pos = 5;

    Tensor<float> dummy(arena, infer.k_state.shape);
    for(int i=0;i<dummy.numel;i++){
        dummy.data[i] = float(i);
    }

    infer.k_state.copy_from(dummy);
    infer.v_state.copy_from(dummy);

    infer.push_kv(0);

    for (size_t h=0; h<infer.config.n_kv_heads; h++){
        if (!equals(infer.k_cache.at({0, h, infer.pos}), dummy.at({h}))){
            std::cout << "KV Cache push k mismatch" << std::endl;
            return 1;
        }

        if (!equals(infer.v_cache.at({0, h, infer.pos}), dummy.at({h}))){
            std::cout << "KV Cache push v mismatch" << std::endl;
            return 1;
        }
    }

    return 0;
}

RegisterTest kv_cache_reg("test kv cache", "f32", &test_kv_cache);

int test_embedding() {
    std::shared_ptr<Parameters> params = get_params();

    Embedding emb(params->get_tensor<float>(-1, "model.embed_tokens.weight"));

    size_t token_id = 0;
    emb.forward(infer, token_id);

    Tensor emb1 = infer.hidden_state;

    if (!equals(emb1.data[0], -2.1864e-36f)) {
        std::cout << "emb1[0][0] mismatch" << std::endl;
        return 1;
    }
    if (!equals(emb1.data[4095], -6.3947e-36f)) {
        std::cout << "emb1[0][-1] mismatch" << std::endl;
        return 1;
    }

    token_id = 31999;
    emb.forward(infer, token_id);
    Tensor emb2 = infer.hidden_state;

    if (!equals(emb2.data[0], -0.0040f)) {
        std::cout << "emb2[-1][0] mismatch" << std::endl;
        return 1;
    }
    if (!equals(emb2.data[4095], -0.0025f)) {
        std::cout << "emb2[-1][-1] mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest embedding_reg("test embedding", "f32", &test_embedding);

int test_rotary_embedding_inv_freq(){
    std::shared_ptr<Parameters> params = get_params();

    RotaryEmbedding::init_freq(infer, params->config);

    if (!equals(expected.at("inv_freq"), infer.inv_freq)){
        std::cout << "Rotary embedding inv freq mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest rotary_embedding_inv_freq_reg("test rotary embedding inv freq", "f32", &test_rotary_embedding_inv_freq);

int test_rotary_embedding(){
    std::shared_ptr<Parameters> params = get_params();

    RotaryEmbedding::init_freq(infer, params->config);

    infer.pos = 0;
    RotaryEmbedding::forward(infer);

    if (!equals(infer.cos, expected.at("cos0"))){
        std::cout << "rotary embedding cos mismatch pos 0" << std::endl;
        return 1;
    }

    if (!equals(infer.sin, expected.at("sin0"))){
        std::cout << "rotary embedding sin mismatch pos 0" << std::endl;
        return 1;
    }

    infer.pos = 3;
    RotaryEmbedding::forward(infer);

    if (!equals(infer.cos, expected.at("cos3"))){
        std::cout << "rotary embedding cos mismatch pos 3" << std::endl;
        return 1;
    }

    if (!equals(infer.sin, expected.at("sin3"))){
        std::cout << "rotary embedding sin mismatch pos 3" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest rotary_embeddingg("test rotary embedding", "f32", &test_rotary_embedding);



int test_rmsnorm() {
    infer.hidden_state.copy_from(expected.at("norm_x"));
    Tensor g = expected.at("norm_g");
    Tensor y = expected.at("norm_y");

    RMSNorm rms(g);
    rms.forward(infer);

    if (!equals(infer.hidden_state, y)) {
        std::cout << "RMSNorm mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest rmsnorm_reg("test rmsnorm", "f32", &test_rmsnorm);


int test_lm_head() {
    std::shared_ptr<Parameters> params = get_params();

    LMHead l(params);
    infer.hidden_state.copy_from(expected.at("lmhead_x"));

    l.forward(infer);

    if (!equals(infer.logits.data[0], 0.0362)) {
        std::cout << "lm head mismatch 1. expected=0.0362 got="
                  << infer.logits.data[0] << std::endl;
        return 1;
    }

    if (!equals(infer.logits.data[31999], -0.0254)) {
        std::cout << "lm head mismatch 2. expected=-0.0254 got="
                  << infer.logits.data[31999] << std::endl;
        return 1;
    }

    return 0;
}


RegisterTest lm_head_reg("test lm head", "f32", &test_lm_head);


