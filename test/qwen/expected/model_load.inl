// Hand-maintained expected values for model-load tests.

const std::vector<const char*> kLayerTensorNames = {
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
};

const std::vector<const char*> kGlobalTensorNames = {
    "model.embed_tokens.weight",
    "model.norm.weight",
};

const std::vector<std::pair<const char*, std::vector<size_t>>> kLayerShapeSpecs = {
    {"input_layernorm.weight", {1024}},
    {"post_attention_layernorm.weight", {1024}},
    {"self_attn.q_norm.weight", {128}},
    {"self_attn.k_norm.weight", {128}},
    {"self_attn.q_proj.weight", {2048, 1024}},
    {"self_attn.k_proj.weight", {1024, 1024}},
    {"self_attn.v_proj.weight", {1024, 1024}},
    {"self_attn.o_proj.weight", {1024, 2048}},
    {"mlp.gate_proj.weight", {3072, 1024}},
    {"mlp.up_proj.weight", {3072, 1024}},
    {"mlp.down_proj.weight", {1024, 3072}},
};

const std::vector<std::pair<const char*, std::vector<size_t>>> kGlobalShapeSpecs = {
    {"model.embed_tokens.weight", {151936, 1024}},
    {"model.norm.weight", {1024}},
};

struct WeightSpotCheck {
    const char* key;
    int layer;
    float first;
    float last;
};

const std::vector<WeightSpotCheck> kWeightSpotChecks = {
    {"model.embed_tokens.weight", -1, -0.00927734375f, 0.00567626953125f},
    {"model.norm.weight", -1, 3.9375f, 3.640625f},
    {"input_layernorm.weight", 0, 0.1357421875f, 0.1552734375f},
    {"post_attention_layernorm.weight", 0, 0.474609375f, 0.5703125f},
    {"self_attn.q_proj.weight", 0, 0.0034027099609375f, 0.07763671875f},
    {"self_attn.k_proj.weight", 0, -0.0166015625f, -0.006072998046875f},
    {"self_attn.v_proj.weight", 0, 0.01214599609375f, -0.0283203125f},
    {"self_attn.o_proj.weight", 0, 0.01446533203125f, -0.0228271484375f},
    {"self_attn.q_norm.weight", 0, 4.53125f, 2.390625f},
    {"self_attn.k_norm.weight", 0, 1.296875f, 2.015625f},
    {"mlp.gate_proj.weight", 0, -0.0186767578125f, 0.01348876953125f},
    {"mlp.up_proj.weight", 0, -0.04248046875f, 0.0162353515625f},
    {"mlp.down_proj.weight", 0, 0.02099609375f, 0.00677490234375f},
    {"input_layernorm.weight", 27, 17.625f, 13.3125f},
    {"self_attn.q_proj.weight", 27, -0.034423828125f, 0.01019287109375f},
    {"mlp.down_proj.weight", 27, -0.01434326171875f, 0.01361083984375f},
};
