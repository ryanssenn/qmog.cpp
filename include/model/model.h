#include "loader/model_load.h"
#include "model/inference_state.h"


// https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
struct Embedding {
    Tensor table;
    size_t embedding_dim;
    Embedding(const Tensor& table) : table(table), embedding_dim(table.shape[1]) {}
    void forward(InferenceState& infer, size_t token_id);
};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L268
struct RotaryEmbedding {
    static void forward(InferenceState& infer);
};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L58
struct RMSNorm {
    Tensor g;
    float e;

    RMSNorm(const Tensor& g, float e) : g(g), e(e) {}
    void forward(Tensor& x);
};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py#L214
struct Attention {
    Tensor q_proj;
    Tensor k_proj;
    Tensor v_proj;
    Tensor o_proj;
    RMSNorm q_norm;
    RMSNorm k_norm;

    size_t layer;

    Attention(const Tensor& q_proj,
              const Tensor& k_proj,
              const Tensor& v_proj,
              const Tensor& o_proj,
              const Tensor& q_norm_weight,
              const Tensor& k_norm_weight,
              float norm_eps,
              size_t layer)
            : q_proj(q_proj),
              k_proj(k_proj),
              v_proj(v_proj),
              o_proj(o_proj),
              q_norm(q_norm_weight, norm_eps),
              k_norm(k_norm_weight, norm_eps),
              layer(layer){}

    void forward(InferenceState& infer);
};


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L35
struct MLP {
    Tensor down_proj;
    Tensor gate_proj;
    Tensor up_proj;

    MLP(const Tensor& down_proj, const Tensor& gate_proj, const Tensor& up_proj)
        : down_proj(down_proj), gate_proj(gate_proj), up_proj(up_proj) {}

    void forward(InferenceState& infer);
};


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L206
struct Layer {
    int i;
    RMSNorm input_norm;
    RMSNorm output_norm;
    Attention attn;
    MLP mlp;

    Layer(int i, std::shared_ptr<ModelLoad> p) :
                                i(i),

                                input_norm(p->get_tensor(i, "input_layernorm.weight"), p->config.norm_eps),
                                output_norm(p->get_tensor(i, "post_attention_layernorm.weight"), p->config.norm_eps),

                                attn(p->get_tensor(i, "self_attn.q_proj.weight"),
                                     p->get_tensor(i, "self_attn.k_proj.weight"),
                                     p->get_tensor(i, "self_attn.v_proj.weight"),
                                     p->get_tensor(i, "self_attn.o_proj.weight"),
                                     p->get_tensor(i, "self_attn.q_norm.weight"),
                                     p->get_tensor(i, "self_attn.k_norm.weight"),
                                     p->config.norm_eps,
                                     i),

                                mlp(p->get_tensor(i, "mlp.down_proj.weight"),
                                    p->get_tensor(i, "mlp.gate_proj.weight"),
                                    p->get_tensor(i, "mlp.up_proj.weight"))
                                {}


    void forward(InferenceState& infer);
};


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L414
struct LMHead {
    Tensor lm_head;

    LMHead(std::shared_ptr<ModelLoad> params)
        : lm_head(params->config.tie_word_embeddings
                      ? params->get_tensor(-1, "model.embed_tokens.weight")
                      : params->get_tensor(-1, "lm_head.weight")) {}

    void forward(InferenceState& infer);
};


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L334
struct Model {
    Embedding embedding;
    RMSNorm norm;
    LMHead lmHead;
    std::vector<Layer> layers;

    Model(std::shared_ptr<ModelLoad> params)
        : embedding(params->get_tensor(-1, "model.embed_tokens.weight")),
          norm(params->get_tensor(-1, "model.norm.weight"), params->config.norm_eps),
          lmHead(params) {
        for (int i=0;i<params->config.n_layers; i++){
            layers.emplace_back(i, params);
        }
    }

    void forward(InferenceState& infer, size_t token_id);
};


