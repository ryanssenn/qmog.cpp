#include "kernels.h"
#include "parameters.h"
#include "inference_state.h"


// https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
struct Embedding {
    Tensor<float> table;
    size_t num_embeddings;
    size_t embedding_dim;
    Embedding(const Tensor<float>& table) : table(table), num_embeddings(table.shape[0]), embedding_dim(table.shape[1]) {}
    void forward(InferenceState& infer, size_t token_id);
};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L268
struct RotaryEmbedding {
    static void init_freq(InferenceState& infer, Config& config);
    static void forward(InferenceState& infer);
};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L58
struct RMSNorm {
    Tensor<float> g;
    float e = 1e-5f;

    RMSNorm(const Tensor<float>& g) : g(g) {}
    void forward(InferenceState& infer);
};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L123
struct Attention {
    Tensor<float> q_proj;
    Tensor<float> k_proj;
    Tensor<float> v_proj;
    Tensor<float> o_proj;

    size_t layer;

    Attention(const Tensor<float>& q_proj,
              const Tensor<float>& k_proj,
              const Tensor<float>& v_proj,
              const Tensor<float>& o_proj,
              size_t layer)
            : q_proj(q_proj),
              k_proj(k_proj),
              v_proj(v_proj),
              o_proj(o_proj),
              layer(layer){}

    void forward(InferenceState& infer);
};


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L35
template <typename TGateUp>
struct MLP {
    Tensor<float> down_proj;
    Tensor<TGateUp> gate_proj;
    Tensor<TGateUp> up_proj;

    MLP(const Tensor<float>& down_proj, const Tensor<TGateUp>& gate_proj, const Tensor<TGateUp>& up_proj)
        : down_proj(down_proj), gate_proj(gate_proj), up_proj(up_proj) {}

    void forward(InferenceState& infer);
};


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L206
template <typename TMlp>
struct Layer {
    int i;
    RMSNorm input_norm;
    RMSNorm output_norm;
    Attention attn;
    MLP<TMlp> mlp;

    Layer(int i, std::shared_ptr<Parameters> p) :
                                i(i),

                                input_norm(p->get_tensor<float>(i, "input_layernorm.weight")),
                                output_norm(p->get_tensor<float>(i, "post_attention_layernorm.weight")),

                                attn(p->get_tensor<float>(i, "self_attn.q_proj.weight"),
                                     p->get_tensor<float>(i, "self_attn.k_proj.weight"),
                                     p->get_tensor<float>(i, "self_attn.v_proj.weight"),
                                     p->get_tensor<float>(i, "self_attn.o_proj.weight"), i),

                                mlp(p->get_tensor<float>(i, "mlp.down_proj.weight"),
                                    p->get_tensor<TMlp>(i, "mlp.gate_proj.weight"),
                                    p->get_tensor<TMlp>(i, "mlp.up_proj.weight"))
                                {}


    void forward(InferenceState& infer);
};


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L414
struct LMHead {
    Tensor<float> lm_head; // [4096, vocab_size]

    LMHead(std::shared_ptr<Parameters> params) : lm_head(params->get_tensor<float>(-1, "lm_head.weight")) {}

    void forward(InferenceState& infer);
};


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L334
template <typename TMlp>
struct Model {
    Embedding embedding;
    RMSNorm norm;
    LMHead lmHead;
    std::vector<Layer<TMlp>> layers;

    Model(std::shared_ptr<Parameters> params) : embedding(params->get_tensor<float>(-1, "model.embed_tokens.weight")), norm(params->get_tensor<float>(-1, "model.norm.weight")), lmHead(params){
        for (int i=0;i<params->config.n_layers; i++){
            layers.emplace_back(i, params);
        }
    }

    void forward(InferenceState& infer, size_t token_id);
};




