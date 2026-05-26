#include "hy_mt2/acl_context.h"
#include "hy_mt2/language_model.h"
#include "hy_mt2/tensor.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

uint16_t f32_to_f16_bits(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    const uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

hy_mt2::Tensor filled(const std::vector<int64_t>& shape, float value) {
    size_t n = 1;
    for (int64_t d : shape) n *= static_cast<size_t>(d);
    std::vector<uint16_t> host(n, f32_to_f16_bits(value));
    hy_mt2::Tensor t(shape, hy_mt2::DType::Float16);
    t.copy_from_host(host.data(), host.size() * sizeof(uint16_t));
    return t;
}

hy_mt2::Tensor patterned(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (int64_t d : shape) n *= static_cast<size_t>(d);
    std::vector<uint16_t> host(n);
    for (size_t i = 0; i < n; ++i) host[i] = f32_to_f16_bits(static_cast<float>((i % 17) - 8) * 0.001f);
    hy_mt2::Tensor t(shape, hy_mt2::DType::Float16);
    t.copy_from_host(host.data(), host.size() * sizeof(uint16_t));
    return t;
}

}  // namespace

int main(int argc, char** argv) {
    int iters = 5;
    int64_t chunk_vocab = 8192;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
        else if (arg == "--chunk" && i + 1 < argc) chunk_vocab = std::stoll(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: " << argv[0] << " [--iters N] [--chunk N]\n";
            return 0;
        }
    }

    hy_mt2::AclContext ctx(0);
    auto cfg = hy_mt2::default_hy_mt2_config();
    hy_mt2::LanguageModelWeights weights;
    weights.final_norm_w = filled({cfg.hidden_size}, 1.0f);
    weights.embed = hy_mt2::Tensor({cfg.vocab_size, cfg.hidden_size}, hy_mt2::DType::Float16);

    std::vector<uint16_t> chunk_host(static_cast<size_t>(cfg.hidden_size) * chunk_vocab);
    for (int64_t start = 0; start < cfg.vocab_size; start += chunk_vocab) {
        const int64_t valid = std::min<int64_t>(chunk_vocab, cfg.vocab_size - start);
        std::fill(chunk_host.begin(), chunk_host.end(), f32_to_f16_bits(0.0f));
        for (int64_t h = 0; h < cfg.hidden_size; ++h) {
            for (int64_t v = 0; v < valid; ++v) {
                chunk_host[static_cast<size_t>(h) * chunk_vocab + v] =
                    f32_to_f16_bits(static_cast<float>(((h * 13 + (start + v) * 7) % 23) - 11) * 0.001f);
            }
        }
        hy_mt2::LmHeadChunk chunk;
        chunk.start_vocab = start;
        chunk.valid_vocab = valid;
        chunk.weight = hy_mt2::Tensor({cfg.hidden_size, chunk_vocab}, hy_mt2::DType::Float16);
        chunk.weight.copy_from_host(chunk_host.data(), chunk_host.size() * sizeof(uint16_t));
        weights.lm_head_chunks.push_back(std::move(chunk));
    }

    auto hidden = patterned({1, cfg.hidden_size});
    (void)hy_mt2::lm_head_greedy(hidden, weights, cfg, ctx.stream());
    const auto begin = std::chrono::steady_clock::now();
    int64_t token = 0;
    for (int i = 0; i < iters; ++i) token = hy_mt2::lm_head_greedy(hidden, weights, cfg, ctx.stream());
    const auto end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - begin).count() / iters;
    std::cout << "chunk_vocab,chunks,iters,ms_per_iter,tok_per_s,last_token\n"
              << chunk_vocab << ',' << weights.lm_head_chunks.size() << ',' << iters << ','
              << ms << ',' << (1000.0 / ms) << ',' << token << '\n';
    return 0;
}
