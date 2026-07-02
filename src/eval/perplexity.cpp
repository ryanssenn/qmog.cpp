#include "eval/perplexity.h"

#include "backend/metal/kernels.h"
#include "loader/model_load.h"
#include "model/inference_state.h"
#include "model/model.h"

#include <cmath>
#include <iostream>

// Teacher-forced perplexity over the prompt tokens using the actual engine.
// PPL = exp( mean_i -log softmax(logits_i)[token_{i+1}] ).
void run_perplexity(std::shared_ptr<ModelLoad> params, InferenceState& infer, const std::vector<uint32_t>& tokens) {
    Model model(params);
    infer.pos = 0;

    double nll = 0.0;
    size_t count = 0;
    std::vector<double> per_token_nll;

    float* logits = infer.logits.f32();
    float* probs = infer.probs.f32();

    for (size_t i = 0; i + 1 < tokens.size(); i++) {
        model.forward(infer, tokens[i]);

        for (size_t j = 0; j < infer.logits.numel; j++) {
            probs[j] = logits[j];
        }
        softmax(infer.probs, infer.probs);

        double logp = std::log(std::max(static_cast<double>(probs[tokens[i + 1]]), 1e-30));
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
