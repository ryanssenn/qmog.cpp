#include "setup/context.h"
#include <cmath>
#include <iomanip>
#include <unordered_map>

static constexpr size_t LOGITS_TOPK = 10;
static constexpr size_t LOGITS_GEN_STEPS = 5;
static constexpr float LAYER_STACK_ATOL = 5e-2f;

static const std::unordered_map<std::string, std::string> PROMPTS = {
    {"sky", "The color of the sky is"},
    {"paris", "Paris is the capital of"},
};

// Count how many of the int8 top-k token ids also appear in the f32 golden top-k.
static size_t topk_overlap(const TopK& got, const Tensor<float>& exp_ids) {
    size_t overlap = 0;
    for (uint32_t id : got.ids) {
        for (size_t j = 0; j < exp_ids.numel; j++) {
            if (static_cast<uint32_t>(exp_ids.data[j]) == id) {
                overlap++;
                break;
            }
        }
    }
    return overlap;
}

// Max abs logit error over ranks whose token id matches the golden at the same rank.
static float aligned_value_error(const TopK& got, const Tensor<float>& exp_ids,
                                 const Tensor<float>& exp_vals) {
    float max_err = 0.0f;
    for (size_t i = 0; i < got.ids.size() && i < exp_ids.numel; i++) {
        if (got.ids[i] == static_cast<uint32_t>(exp_ids.data[i])) {
            max_err = std::max(max_err, std::fabs(got.vals[i] - exp_vals.data[i]));
        }
    }
    return max_err;
}

struct DriftStats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float rmse = 0.0f;
    float cosine = 0.0f;
    float got_rms = 0.0f;
    float exp_rms = 0.0f;
    size_t first_bad = static_cast<size_t>(-1);
};

static DriftStats measure_drift(const Tensor<float>& got, const Tensor<float>& exp) {
    DriftStats s;
    if (got.numel != exp.numel || got.numel == 0) {
        return s;
    }

    double abs_sum = 0.0;
    double sq_sum = 0.0;
    double dot = 0.0;
    double got_norm = 0.0;
    double exp_norm = 0.0;

    for (size_t i = 0; i < got.numel; i++) {
        double g = got.data[i];
        double e = exp.data[i];
        double diff = g - e;
        double ad = std::fabs(diff);

        if (ad > s.max_abs) {
            s.max_abs = static_cast<float>(ad);
        }
        if (s.first_bad == static_cast<size_t>(-1) && ad >= LAYER_STACK_ATOL) {
            s.first_bad = i;
        }

        abs_sum += ad;
        sq_sum += diff * diff;
        dot += g * e;
        got_norm += g * g;
        exp_norm += e * e;
    }

    s.mean_abs = static_cast<float>(abs_sum / got.numel);
    s.rmse = static_cast<float>(std::sqrt(sq_sum / got.numel));
    s.got_rms = static_cast<float>(std::sqrt(got_norm / got.numel));
    s.exp_rms = static_cast<float>(std::sqrt(exp_norm / got.numel));

    double denom = std::sqrt(got_norm) * std::sqrt(exp_norm);
    if (denom > 0.0) {
        s.cosine = static_cast<float>(dot / denom);
    }

    return s;
}

static void print_drift(const std::string& prefix, const std::string& label, const DriftStats& s) {
    std::cout << "  [" << prefix << "] " << std::setw(6) << label
              << " max_abs=" << s.max_abs
              << " mean_abs=" << s.mean_abs
              << " rmse=" << s.rmse
              << " cosine=" << s.cosine;
    if (s.exp_rms > 0.0f) {
        std::cout << " rms_ratio=" << (s.got_rms / s.exp_rms);
    }
    if (s.first_bad != static_cast<size_t>(-1)) {
        std::cout << " first_bad=" << s.first_bad;
    }
    std::cout << std::endl;
}

static float token_nll(const Tensor<float>& logits, uint32_t target) {
    float max_logit = logits.data[0];
    for (size_t i = 1; i < logits.numel; i++) {
        if (logits.data[i] > max_logit) {
            max_logit = logits.data[i];
        }
    }

    double sum = 0.0;
    for (size_t i = 0; i < logits.numel; i++) {
        sum += std::exp(static_cast<double>(logits.data[i] - max_logit));
    }

    float logsumexp = max_logit + static_cast<float>(std::log(sum));
    return logsumexp - logits.data[target];
}

template <typename TMlp>
static int run_ppl_prompt(const std::string& prefix) {
    std::shared_ptr<Parameters> params = get_params();
    Model<TMlp> model(params);

    auto it = PROMPTS.find(prefix);
    if (it == PROMPTS.end()) {
        std::cout << "Unknown perplexity prompt prefix: " << prefix << std::endl;
        return 1;
    }

    std::string nll_key = "ppl_" + prefix + "_token_nll";
    std::string mean_key = "ppl_" + prefix + "_mean_nll";
    std::string ppl_key = "ppl_" + prefix + "_ppl";
    if (expected.find(nll_key) == expected.end() ||
        expected.find(mean_key) == expected.end() ||
        expected.find(ppl_key) == expected.end()) {
        std::cout << "Skipping perplexity for " << prefix
                  << " (regenerate logits goldens)" << std::endl;
        return 0;
    }

    std::vector<uint32_t> tokens = params->tokenizer.encode(it->second);
    infer.pos = 0;

    double total_nll = 0.0;
    float max_token_delta = 0.0f;
    const Tensor<float>& exp_nll = expected.at(nll_key);

    for (size_t i = 0; i + 1 < tokens.size(); i++) {
        model.forward(infer, tokens[i]);
        float nll = token_nll(infer.logits, tokens[i + 1]);
        total_nll += nll;

        if (i < exp_nll.numel) {
            float delta = std::fabs(nll - exp_nll.data[i]);
            if (delta > max_token_delta) {
                max_token_delta = delta;
            }
        }
    }

    size_t count = tokens.size() > 0 ? tokens.size() - 1 : 0;
    if (count == 0) {
        return 0;
    }

    float mean_nll = static_cast<float>(total_nll / count);
    float ppl = std::exp(mean_nll);
    float ref_mean_nll = expected.at(mean_key).data[0];
    float ref_ppl = expected.at(ppl_key).data[0];

    std::cout << "  [" << prefix << "]"
              << " tokens=" << count
              << " mean_nll f32=" << ref_mean_nll
              << " int8=" << mean_nll
              << " delta=" << (mean_nll - ref_mean_nll)
              << " ppl f32=" << ref_ppl
              << " int8=" << ppl
              << " max_token_nll_delta=" << max_token_delta
              << std::endl;

    return std::isfinite(mean_nll) && std::isfinite(ppl) ? 0 : 1;
}

template <typename TMlp>
static int test_prompt_ppl() {
    if (!has_logits_golden()) {
        std::cout << "Skipping perplexity tests (missing test/mistral/logits_expected.txt). "
                  << "Run: python scripts/test/mistral/logits.py" << std::endl;
        return 0;
    }

    int failed = 0;
    for (const auto& entry : PROMPTS) {
        if (run_ppl_prompt<TMlp>(entry.first) != 0) {
            failed = 1;
        }
    }
    return failed;
}

// Teacher-forced divergence report: at every step we feed the f32 golden's top-1
// token, so the int8 model always sees the exact context f32 saw. This isolates
// per-step int8 logit error instead of letting trajectories drift apart.
// Pass/fail is decided by top-1 argmax agreement (what greedy generation uses);
// logit value gaps are printed as diagnostics only.
template <typename TMlp>
static int run_logits_prompt(const std::string& prefix) {
    std::shared_ptr<Parameters> params = get_params();
    Model<TMlp> model(params);

    auto it = PROMPTS.find(prefix);
    if (it == PROMPTS.end()) {
        std::cout << "Unknown logits prompt prefix: " << prefix << std::endl;
        return 1;
    }

    std::vector<uint32_t> tokens = params->tokenizer.encode(it->second);
    infer.pos = 0;

    for (size_t i = 0; i + 1 < tokens.size(); i++) {
        model.forward(infer, tokens[i]);
    }

    int failed = 0;
    uint32_t t = tokens.back();
    for (size_t step = 0; step <= LOGITS_GEN_STEPS; step++) {
        model.forward(infer, t);

        TopK got = get_topk(infer.logits, LOGITS_TOPK);
        std::string ids_key = "logits_" + prefix + "_step" + std::to_string(step) + "_top10_ids";
        std::string vals_key = "logits_" + prefix + "_step" + std::to_string(step) + "_top10_vals";

        if (expected.find(ids_key) == expected.end() || expected.find(vals_key) == expected.end()) {
            std::cout << "Missing golden keys for " << prefix << " step " << step << std::endl;
            return 1;
        }

        const Tensor<float>& exp_ids = expected.at(ids_key);
        const Tensor<float>& exp_vals = expected.at(vals_key);

        uint32_t exp_top1 = static_cast<uint32_t>(exp_ids.data[0]);
        bool top1_match = !got.ids.empty() && got.ids[0] == exp_top1;
        size_t overlap = topk_overlap(got, exp_ids);
        float max_err = aligned_value_error(got, exp_ids, exp_vals);

        std::cout << "  [" << prefix << "] step " << step
                  << " top1 f32=" << exp_top1 << "(" << exp_vals.data[0] << ")"
                  << " int8=" << (got.ids.empty() ? 0 : got.ids[0])
                  << "(" << (got.vals.empty() ? 0.0f : got.vals[0]) << ")"
                  << " | top1=" << (top1_match ? "OK" : "FLIP")
                  << " top10_overlap=" << overlap << "/" << LOGITS_TOPK
                  << " max_val_err=" << max_err
                  << std::endl;

        if (!top1_match) {
            failed = 1;
        }

        if (step == LOGITS_GEN_STEPS) {
            break;
        }

        // Teacher forcing: follow the golden trajectory, not int8's own pick.
        t = exp_top1;
    }

    return failed;
}

template <typename TMlp>
static int test_logits_multi() {
    if (!has_logits_golden()) {
        std::cout << "Skipping logits tests (missing test/mistral/logits_expected.txt). "
                  << "Run: python scripts/test/mistral/logits.py" << std::endl;
        return 0;
    }

    int failed = 0;
    for (const auto& entry : PROMPTS) {
        if (run_logits_prompt<TMlp>(entry.first) != 0) {
            failed = 1;
        }
    }
    return failed;
}

template <typename TMlp>
static int run_layer_stack_prompt(const std::string& prefix) {
    std::shared_ptr<Parameters> params = get_params();
    Model<TMlp> model(params);

    auto it = PROMPTS.find(prefix);
    if (it == PROMPTS.end()) {
        return 1;
    }

    std::vector<uint32_t> tokens = params->tokenizer.encode(it->second);
    infer.pos = 0;

    for (size_t i = 0; i + 1 < tokens.size(); i++) {
        model.forward(infer, tokens[i]);
    }

    model.embedding.forward(infer, tokens.back());

    int failed = 0;

    for (size_t layer = 0; layer < model.layers.size(); layer++) {
        model.layers[layer].forward(infer);

        std::string key = "layer_stack_" + prefix + "_L" + std::to_string(layer);
        if (expected.find(key) == expected.end()) {
            std::cout << "Missing golden key: " << key << std::endl;
            return 1;
        }

        DriftStats drift = measure_drift(infer.hidden_state, expected.at(key));
        print_drift(prefix, "L" + std::to_string(layer), drift);
        if (drift.first_bad != static_cast<size_t>(-1)) {
            failed = 1;
        }
    }

    model.norm.forward(infer);
    std::string norm_key = "layer_stack_" + prefix + "_norm";
    DriftStats drift = measure_drift(infer.hidden_state, expected.at(norm_key));
    print_drift(prefix, "norm", drift);
    if (drift.first_bad != static_cast<size_t>(-1)) {
        failed = 1;
    }

    return failed;
}

template <typename TMlp>
static int test_layer_stack() {
    if (expected.find("layer_stack_sky_L0") == expected.end()) {
        std::cout << "Skipping layer stack tests (regenerate with DUMP_LAYER_STACK=1). "
                  << "Run: DUMP_LAYER_STACK=1 python scripts/test/mistral/logits.py" << std::endl;
        return 0;
    }

    int failed = 0;
    for (const auto& entry : PROMPTS) {
        if (run_layer_stack_prompt<TMlp>(entry.first) != 0) {
            failed = 1;
        }
    }
    return failed;
}

static int test_logits_multi_f32() { return test_logits_multi<float>(); }
static int test_logits_multi_int8() { return test_logits_multi<int8_t>(); }
static int test_prompt_ppl_f32() { return test_prompt_ppl<float>(); }
static int test_prompt_ppl_int8() { return test_prompt_ppl<int8_t>(); }
static int test_layer_stack_f32() { return test_layer_stack<float>(); }
static int test_layer_stack_int8() { return test_layer_stack<int8_t>(); }

RegisterTest logits_multi_reg("test logits multi top10", "f32", &test_logits_multi_f32);
RegisterTest logits_multi_reg_int8("test logits multi top10", "int8", &test_logits_multi_int8);
RegisterTest prompt_ppl_reg("test prompt perplexity", "f32", &test_prompt_ppl_f32);
RegisterTest prompt_ppl_reg_int8("test prompt perplexity", "int8", &test_prompt_ppl_int8);
RegisterTest layer_stack_reg("test layer stack prefill", "f32", &test_layer_stack_f32);
RegisterTest layer_stack_reg_int8("test layer stack prefill", "int8", &test_layer_stack_int8);
