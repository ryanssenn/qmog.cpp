#include "setup/context.h"
#include "fp16.h"

template <typename TGateUp, typename TLinear>
int test_layer() {
    std::shared_ptr<Parameters> params = get_params();
    infer.pos = 0;

    Layer<TGateUp, TLinear> layer(0, params);

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

template <typename TLinear>
int test_attention() {
    std::shared_ptr<Parameters> params = get_params();
    infer.pos = 0;

    Attention<TLinear> attn(params->get_tensor<TLinear>(0, "self_attn.q_proj.weight"), params->get_tensor<TLinear>(0, "self_attn.k_proj.weight"), params->get_tensor<TLinear>(0, "self_attn.v_proj.weight"), params->get_tensor<TLinear>(0, "self_attn.o_proj.weight"), 0);

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

        if (!equals(infer.hidden_state, expected.at("attn_o" + std::to_string(i)))){
            std::cout << "Attention result mismatch at token " + std::to_string(i)  << std::endl;
            return 1;
        }
        infer.pos++;
    }

    return 0;
}

static int test_attention_dispatch() {
    auto params = get_params();
    if (params->uses_f16_linear_weights()) {
        return test_attention<fp16_t>();
    }
    return test_attention<float>();
}

RegisterTest attention_reg("test attention", "any", &test_attention_dispatch);

template <typename TGateUp, typename TLinear>
int test_mlp(){
    std::shared_ptr<Parameters> params = get_params();

    MLP<TGateUp, TLinear> mlp(params->get_tensor<TLinear>(0, "mlp.down_proj.weight"),
               params->get_tensor<TGateUp>(0, "mlp.gate_proj.weight"),
               params->get_tensor<TGateUp>(0, "mlp.up_proj.weight"));

    infer.hidden_state.copy_from(expected.at("mlp_h"));

    mlp.forward(infer);

    if (!equals(infer.hidden_state, expected.at("mlp_output"))){
        std::cout << "MLP feedforward result mismatch" << std::endl;
        return 1;
    }

    return 0;
}

static int test_mlp_int8() {
    auto params = get_params();
    if (params->uses_f16_linear_weights()) {
        return test_mlp<int8_t, fp16_t>();
    }
    return test_mlp<int8_t, float>();
}
RegisterTest mlp_feedforward_reg_q("test attention feedforward mlp", "Q8F16", &test_mlp_int8);

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
        Tensor<fp16_t> k_cache_head = infer.k_cache.at({0, h, infer.pos});
        Tensor<fp16_t> v_cache_head = infer.v_cache.at({0, h, infer.pos});
        Tensor<float> expected_head = dummy.at({h});

        for (size_t i = 0; i < expected_head.numel; i++) {
            if (!equals(k_cache_head.get(i), expected_head.data[i])) {
                std::cout << "KV Cache push k mismatch" << std::endl;
                return 1;
            }

            if (!equals(v_cache_head.get(i), expected_head.data[i])) {
                std::cout << "KV Cache push v mismatch" << std::endl;
                return 1;
            }
        }
    }

    return 0;
}

RegisterTest kv_cache_reg("test kv cache", "any", &test_kv_cache);

static int test_embedding_dispatch() {
    std::shared_ptr<Parameters> params = get_params();

    if (params->uses_f16_linear_weights()) {
        Embedding<fp16_t> emb(params->get_tensor<fp16_t>(-1, "model.embed_tokens.weight"));
        size_t token_id = 0;
        emb.forward(infer, token_id);
        if (!equals(infer.hidden_state.data[0], -2.1864e-36f)) {
            std::cout << "emb1[0][0] mismatch" << std::endl;
            return 1;
        }
        if (!equals(infer.hidden_state.data[4095], -6.3947e-36f)) {
            std::cout << "emb1[0][-1] mismatch" << std::endl;
            return 1;
        }
        token_id = 31999;
        emb.forward(infer, token_id);
        if (!equals(infer.hidden_state.data[0], -0.0040f)) {
            std::cout << "emb2[-1][0] mismatch" << std::endl;
            return 1;
        }
        if (!equals(infer.hidden_state.data[4095], -0.0025f)) {
            std::cout << "emb2[-1][-1] mismatch" << std::endl;
            return 1;
        }
        return 0;
    }

    Embedding<float> emb(params->get_tensor<float>(-1, "model.embed_tokens.weight"));
    size_t token_id = 0;
    emb.forward(infer, token_id);
    if (!equals(infer.hidden_state.data[0], -2.1864e-36f)) {
        std::cout << "emb1[0][0] mismatch" << std::endl;
        return 1;
    }
    if (!equals(infer.hidden_state.data[4095], -6.3947e-36f)) {
        std::cout << "emb1[0][-1] mismatch" << std::endl;
        return 1;
    }
    token_id = 31999;
    emb.forward(infer, token_id);
    if (!equals(infer.hidden_state.data[0], -0.0040f)) {
        std::cout << "emb2[-1][0] mismatch" << std::endl;
        return 1;
    }
    if (!equals(infer.hidden_state.data[4095], -0.0025f)) {
        std::cout << "emb2[-1][-1] mismatch" << std::endl;
        return 1;
    }
    return 0;
}

RegisterTest embedding_reg("test embedding", "any", &test_embedding_dispatch);

int test_rotary_embedding_inv_freq(){
    std::shared_ptr<Parameters> params = get_params();

    RotaryEmbedding::init_freq(infer, params->config);

    if (!equals(expected.at("inv_freq"), infer.inv_freq)){
        std::cout << "Rotary embedding inv freq mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest rotary_embedding_inv_freq_reg("test rotary embedding inv freq", "any", &test_rotary_embedding_inv_freq);

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

RegisterTest rotary_embeddingg("test rotary embedding", "any", &test_rotary_embedding);



int test_rmsnorm() {
    infer.hidden_state.copy_from(expected.at("norm_x"));
    Tensor g = expected.at("norm_g");
    Tensor y = expected.at("norm_y");

    RMSNorm<float> rms(g);
    rms.forward(infer);

    if (!equals(infer.hidden_state, y)) {
        std::cout << "RMSNorm mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest rmsnorm_reg("test rmsnorm", "any", &test_rmsnorm);


static int test_lm_head_dispatch() {
    std::shared_ptr<Parameters> params = get_params();

    if (params->uses_f16_linear_weights()) {
        LMHead<fp16_t> l(params);
        infer.hidden_state.copy_from(expected.at("lmhead_x"));
        l.forward(infer);
    } else {
        LMHead<float> l(params);
        infer.hidden_state.copy_from(expected.at("lmhead_x"));
        l.forward(infer);
    }

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


RegisterTest lm_head_reg("test lm head", "any", &test_lm_head_dispatch);
