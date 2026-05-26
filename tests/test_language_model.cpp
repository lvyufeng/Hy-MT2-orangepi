#include "hy_mt2/acl_context.h"
#include "hy_mt2/language_model.h"
#include "hy_mt2/tensor.h"

#include <cstdint>
#include <cstring>
#include <iostream>
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

hy_mt2::Tensor make_tensor(const std::vector<int64_t>& shape, const std::vector<float>& values) {
    std::vector<uint16_t> bits(values.size());
    for (size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_f16_bits(values[i]);
    hy_mt2::Tensor t(shape, hy_mt2::DType::Float16);
    t.copy_from_host(bits.data(), bits.size() * sizeof(uint16_t));
    return t;
}

hy_mt2::Tensor filled(const std::vector<int64_t>& shape, float value) {
    size_t n = 1;
    for (int64_t d : shape) n *= static_cast<size_t>(d);
    return make_tensor(shape, std::vector<float>(n, value));
}

hy_mt2::LayerWeights make_zero_layer() {
    hy_mt2::LayerWeights layer;
    layer.input_norm_w = filled({4}, 1.0f);
    layer.post_norm_w = filled({4}, 1.0f);
    layer.q_w = filled({4, 4}, 0.0f);
    layer.k_w = filled({2, 4}, 0.0f);
    layer.v_w = filled({2, 4}, 0.0f);
    layer.o_w = filled({4, 4}, 0.0f);
    layer.q_norm_w = filled({2}, 1.0f);
    layer.k_norm_w = filled({2}, 1.0f);
    layer.gate_w = filled({8, 4}, 0.0f);
    layer.up_w = filled({8, 4}, 0.0f);
    layer.down_w = filled({4, 8}, 0.0f);
    return layer;
}

}  // namespace

int main() {
    hy_mt2::AclContext ctx(0);
    hy_mt2::LanguageModelConfig cfg;
    cfg.hidden_size = 4;
    cfg.intermediate_size = 8;
    cfg.num_q_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.head_dim = 2;
    cfg.num_layers = 1;
    cfg.vocab_size = 4;
    cfg.max_position_embeddings = 8;
    cfg.eos_token_id = 3;

    hy_mt2::LanguageModelWeights w;
    w.embed = make_tensor({4, 4}, {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    });
    w.final_norm_w = filled({4}, 1.0f);
    w.layers.push_back(make_zero_layer());

    hy_mt2::Tensor cos;
    hy_mt2::Tensor sin;
    hy_mt2::build_rope_tables(8, cfg, cos, sin);
    auto state = hy_mt2::make_decode_state(8, cfg, ctx.stream());
    auto last = hy_mt2::prefill({0, 1}, w, cfg, cos, sin, state, ctx.stream());
    if (state.seq_len != 2) {
        std::cerr << "prefill did not advance seq_len\n";
        return 1;
    }
    const int64_t next = hy_mt2::lm_head_greedy(last, w, cfg, ctx.stream());
    if (next != 1) {
        std::cerr << "lm_head expected token 1, got " << next << '\n';
        return 1;
    }
    const int64_t step = hy_mt2::decode_step_greedy(2, w, cfg, cos, sin, state, ctx.stream());
    if (state.seq_len != 3 || step != 2) {
        std::cerr << "decode_step expected token 2 and seq_len 3, got token "
                  << step << " seq_len " << state.seq_len << '\n';
        return 1;
    }

    std::cout << "[ok] language model prefill/decode synthetic path\n";
    return 0;
}
