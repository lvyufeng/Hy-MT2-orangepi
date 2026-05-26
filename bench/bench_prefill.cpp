#include "hy_mt2/acl_context.h"
#include "hy_mt2/language_model.h"
#include "hy_mt2/weights.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string arg_value(int& i, int argc, char** argv, const char* name) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
    return argv[++i];
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir = hy_mt2::default_model_dir();
    int64_t prompt_len = 8;
    int iters = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model") model_dir = arg_value(i, argc, argv, "--model");
        else if (arg == "--prompt-len") prompt_len = std::stoll(arg_value(i, argc, argv, "--prompt-len"));
        else if (arg == "--iters") iters = std::stoi(arg_value(i, argc, argv, "--iters"));
        else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: " << argv[0] << " [--model DIR] [--prompt-len N] [--iters N]\n";
            return 0;
        }
    }

    const std::filesystem::path st = std::filesystem::path(model_dir) / "model.safetensors";
    if (!std::filesystem::exists(st)) {
        std::cout << "[skip] model.safetensors not found at " << st << '\n';
        return 0;
    }

    hy_mt2::AclContext ctx(0);
    auto cfg = hy_mt2::default_hy_mt2_config();
    const int64_t max_seq_len = std::max<int64_t>(prompt_len + 1, 4096);
    hy_mt2::Tensor cos;
    hy_mt2::Tensor sin;
    hy_mt2::build_rope_tables(max_seq_len, cfg, cos, sin);
    hy_mt2::WeightsIndex index(st.string());
    auto weights = hy_mt2::load_language_model_weights(index, cfg);

    std::vector<int32_t> ids(static_cast<size_t>(prompt_len), static_cast<int32_t>(cfg.bos_token_id));
    double total_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
        auto state = hy_mt2::make_decode_state(max_seq_len, cfg, ctx.stream());
        const auto start = std::chrono::steady_clock::now();
        (void)hy_mt2::prefill(ids, weights, cfg, cos, sin, state, ctx.stream());
        const auto end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    const double ms = total_ms / static_cast<double>(iters);
    std::cout << "prompt_len,iters,ms,tok_per_s\n"
              << prompt_len << ',' << iters << ',' << ms << ','
              << (static_cast<double>(prompt_len) * 1000.0 / ms) << '\n';
    return 0;
}
