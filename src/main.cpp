#include <iostream>
#include <string>
#include <vector>

#include "backend/metal/metal_backend.h"
#include "loader/model_load.h"
#include "model/inference_state.h"
#include "model/model.h"
#include "decode/sampling.h"
#include "eval/perplexity.h"

int main(int argc, char** argv) {
    MetalBackend::init();

    float temp = 0.0f;
    bool ppl = false;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--temp" && i + 1 < argc) {
            temp = std::stof(argv[++i]);
        } else if (arg == "--ppl") {
            ppl = true;
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <prompt> [--temp TEMP] [--ppl]\n"
                  << "  --temp  Sampling temperature. 0 = greedy (default).\n"
                  << "  --ppl   Teacher-forced perplexity over the prompt.\n";
        return 1;
    }

    auto params = std::make_shared<ModelLoad>();
    params->load(positional[0]);
    auto tokens = params->tokenizer.encode(positional[1]);

    InferenceState state(params->config);

    if (ppl) {
        run_perplexity(params, state, tokens);
        return 0;
    }

    Model model(params);

    for (size_t step = 0; step < tokens.size(); step++) {
        model.forward(state, tokens[step]);
    }

    for (int i = 0; i < 32; i++) {
        uint32_t next = temp <= 0.0f ? sample_max(state) : sample_multinomial(state, temp);
        std::cout << params->tokenizer.decode({next});
        std::cout.flush();
        if (next == params->config.eos_token_id) break;
        model.forward(state, next);
    }
    std::cout << "\n";
    return 0;
}
