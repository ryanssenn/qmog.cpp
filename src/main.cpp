#include <iostream>
#include "../include/parameters.h"
#include "../include/modules.h"
#include <random>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_set>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> distr(0.0, 1.0);

static constexpr float REPETITION_PENALTY = 1.15f;

uint64_t get_timestamp_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

void apply_repetition_penalty(InferenceState& infer, const std::vector<uint32_t>& history) {
    std::unordered_set<uint32_t> seen;
    for (uint32_t token_id : history) {
        if (!seen.insert(token_id).second) {
            continue;
        }
        float& logit = infer.logits.data[token_id];
        if (logit > 0.0f) {
            logit /= REPETITION_PENALTY;
        } else {
            logit *= REPETITION_PENALTY;
        }
    }
}

uint32_t sample_max(InferenceState& infer){
    float max_val = infer.logits.data[0];
    size_t res = 0;
    for (size_t i = 0;i < infer.config.vocab_size; i++){
        if (infer.logits.data[i] > max_val){
            res = i;
            max_val = infer.logits.data[i];
        }
    }

    return res;
}

uint32_t sample_multinomial(InferenceState& infer, float temp){
    if (temp <= 0) {
        return sample_max(infer);
    }

    for (int i=0; i<infer.logits.numel; i++) {
        infer.probs.data[i] = infer.logits.data[i] / temp;
    }
    softmax(infer.probs, infer.probs);

    float r = distr(gen);
    float total = 0;

    for (int i=0; i<infer.probs.numel; i++){
        total += infer.probs.data[i];
        if (total >= r){
            return i;
        }
    }
    return infer.probs.numel - 1;
}

template <typename T>
uint32_t generate(
    Model<T>& model,
    InferenceState& infer,
    size_t token,
    float temp,
    std::vector<uint32_t>& history
){
    model.forward(infer, token);
    apply_repetition_penalty(infer, history);

    if (temp <= 0) {
        return sample_max(infer);
    }
    return sample_multinomial(infer, temp);
}

template <typename T>
void run_inference(std::shared_ptr<Parameters> params, InferenceState& infer, const std::vector<uint32_t>& got, float temp) {
    Model<T> model(params);

    for (int i=0; i<(int)got.size()-1; i++){
        model.forward(infer, got[i]);
    }

    uint64_t start_time = get_timestamp_ms();

    uint32_t t = got[got.size()-1];
    int i = 0;

    std::vector<uint32_t> history = got;
    
    // Get EOS token ID (</s> for Mistral), fallback to 2 if not found
    uint32_t eos_token_id = 2;  // Default fallback
    auto eos_it = params->tokenizer.token_to_id.find("</s>");
    if (eos_it != params->tokenizer.token_to_id.end()) {
        eos_token_id = eos_it->second;
    }
    
    for (;i<50;i++){
        t = generate(model, infer, t, temp, history);
        history.push_back(t);

        if (t == eos_token_id){
            break;
        }

        std::cout << params->tokenizer.decode({t}) << std::flush;
    }

    std::cout << std::endl;

    uint64_t end_time = get_timestamp_ms();

    std::cout << "throughput: " << (i+1) / ((end_time - start_time) / 1000.0) << " tok/s" << std::endl;
}

// Teacher-forced perplexity over the prompt tokens using the actual engine.
// PPL = exp( mean_i -log softmax(logits_i)[token_{i+1}] ).
template <typename T>
void run_perplexity(std::shared_ptr<Parameters> params, InferenceState& infer, const std::vector<uint32_t>& tokens) {
    Model<T> model(params);
    infer.pos = 0;

    size_t vocab = params->config.vocab_size;
    double nll = 0.0;
    size_t count = 0;
    std::vector<double> per_token_nll;

    for (size_t i = 0; i + 1 < tokens.size(); i++) {
        model.forward(infer, tokens[i]);

        // Numerically stable log-softmax over the vocab.
        float maxv = infer.logits.data[0];
        for (size_t j = 1; j < vocab; j++) {
            maxv = std::max(maxv, infer.logits.data[j]);
        }
        double sumexp = 0.0;
        for (size_t j = 0; j < vocab; j++) {
            sumexp += std::exp((double)infer.logits.data[j] - maxv);
        }
        double log_z = (double)maxv + std::log(sumexp);
        double logp = (double)infer.logits.data[tokens[i + 1]] - log_z;

        nll += -logp;
        per_token_nll.push_back(-logp);
        count++;
    }

    std::cout << "token_ids:";
    for (uint32_t id : tokens) std::cout << " " << id;
    std::cout << std::endl;
    std::cout << "per_token_nll:";
    for (double v : per_token_nll) std::cout << " " << v;
    std::cout << std::endl;
    std::cout << "tokens: " << tokens.size() << std::endl;
    std::cout << "perplexity: " << std::exp(nll / count) << std::endl;
}

int main(int argc, char** argv) {
    float temp = 0.0f;
    bool ppl = false;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--temp" && i + 1 < argc) {
            temp = std::stof(argv[++i]);
        } else if (arg == "--ppl") {
            ppl = true;
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <prompt> [--temp TEMP]" << std::endl;
        std::cerr << "  --temp  Sampling temperature. 0 = greedy (default). Try 0.7 for less repetition." << std::endl;
        return 1;
    }

    std::string model_path = positional[0];
    std::string text = positional[1];

    std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
    params->load_parameters(model_path);

    InferenceState infer(params->config);
    std::vector<uint32_t> got = params->tokenizer.encode(text);

    if (ppl) {
        if (params->config.quant == "int8") {
            run_perplexity<int8_t>(params, infer, got);
        } else if (params->config.quant == "f32") {
            run_perplexity<float>(params, infer, got);
        }
        return 0;
    }

    if (params->config.quant == "int8") {
        run_inference<int8_t>(params, infer, got, temp);
    } else if (params->config.quant == "f32") {
        run_inference<float>(params, infer, got, temp);
    }

    return 0;
}
