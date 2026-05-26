#include "hy_mt2/acl_context.h"
#include "hy_mt2/decoder_layer.h"
#include "hy_mt2/ops.h"
#include "hy_mt2/tensor.h"

#include <algorithm>
#include <cmath>
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

float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
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

std::vector<float> read_fp16(const hy_mt2::Tensor& t) {
    std::vector<uint16_t> bits(t.numel());
    t.copy_to_host(bits.data(), bits.size() * sizeof(uint16_t));
    std::vector<float> out(bits.size());
    for (size_t i = 0; i < bits.size(); ++i) out[i] = f16_bits_to_f32(bits[i]);
    return out;
}

}  // namespace

int main() {
    hy_mt2::AclContext ctx(0);
    const hy_mt2::DecoderLayerConfig cfg{2, 1, 2, 1e-5};
    auto input_norm = filled({4}, 1.0f);
    auto post_norm = filled({4}, 1.0f);
    auto q_w = filled({4, 4}, 0.0f);
    auto k_w = filled({4, 2}, 0.0f);
    auto v_w = filled({4, 2}, 0.0f);
    auto o_w = filled({4, 4}, 0.0f);
    auto q_norm = filled({2}, 1.0f);
    auto k_norm = filled({2}, 1.0f);
    auto gate_w = filled({4, 8}, 0.0f);
    auto up_w = filled({4, 8}, 0.0f);
    auto down_w = filled({8, 4}, 0.0f);

    const hy_mt2::DecoderLayerWeights w{
        &input_norm, &post_norm, &q_w, &k_w, &v_w, &o_w,
        &q_norm, &k_norm, &gate_w, &up_w, &down_w,
    };

    auto cos = filled({4, 1}, 1.0f);
    auto sin = filled({4, 1}, 0.0f);
    auto cache = hy_mt2::make_layer_cache(4, cfg, ctx.stream());

    auto hidden = make_tensor({2, 4}, {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 2.0f, 4.0f, -0.25f});
    hy_mt2::Tensor out({2, 4}, hy_mt2::DType::Float16); out.allocate();
    hy_mt2::decoder_layer_prefill(hidden, w, cos, sin, {0, 1}, cfg, cache, out, ctx.stream());
    const auto prefill_out = read_fp16(out);
    const auto hidden_out = read_fp16(hidden);
    for (size_t i = 0; i < hidden_out.size(); ++i) {
        if (std::fabs(prefill_out[i] - hidden_out[i]) > 5e-3f) {
            std::cerr << "decoder prefill residual mismatch at " << i << ": "
                      << prefill_out[i] << " vs " << hidden_out[i] << '\n';
            return 1;
        }
    }

    if (hy_mt2::has_attention_step_custom()) {
        constexpr int64_t num_q_heads = 2;
        constexpr int64_t num_kv_heads = 1;
        constexpr int64_t head_dim = 128;
        constexpr int64_t context = 3;
        constexpr int64_t max_seq = 4;
        std::vector<float> q_vals(static_cast<size_t>(num_q_heads * head_dim));
        std::vector<float> k_vals(static_cast<size_t>(max_seq * num_kv_heads * head_dim), 0.0f);
        std::vector<float> v_vals(static_cast<size_t>(max_seq * num_kv_heads * head_dim), 0.0f);
        for (size_t i = 0; i < q_vals.size(); ++i) q_vals[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.01f;
        for (size_t i = 0; i < k_vals.size(); ++i) k_vals[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.015f;
        for (size_t i = 0; i < v_vals.size(); ++i) v_vals[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.02f;
        auto q = make_tensor({num_q_heads, head_dim}, q_vals);
        auto k_cache = make_tensor({max_seq, num_kv_heads * head_dim}, k_vals);
        auto v_cache = make_tensor({max_seq, num_kv_heads * head_dim}, v_vals);
        hy_mt2::Tensor custom_out({num_q_heads, head_dim}, hy_mt2::DType::Float16); custom_out.allocate();
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        hy_mt2::attention_step_custom(q, k_cache, v_cache, context, num_q_heads, num_kv_heads,
                                      scale, custom_out, ctx.stream());
        const auto q_host = read_fp16(q);
        const auto k_host = read_fp16(k_cache);
        const auto v_host = read_fp16(v_cache);
        std::vector<float> ref(static_cast<size_t>(num_q_heads * head_dim), 0.0f);
        for (int64_t qh = 0; qh < num_q_heads; ++qh) {
            std::vector<float> scores(static_cast<size_t>(context));
            float max_score = -1.0e30f;
            for (int64_t t = 0; t < context; ++t) {
                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) {
                    dot += q_host[static_cast<size_t>(qh * head_dim + d)] *
                           k_host[static_cast<size_t>(t * head_dim + d)];
                }
                scores[static_cast<size_t>(t)] = dot * scale;
                max_score = std::max(max_score, scores[static_cast<size_t>(t)]);
            }
            float sum = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - max_score);
                sum += score;
            }
            for (int64_t t = 0; t < context; ++t) {
                const float p = scores[static_cast<size_t>(t)] / sum;
                for (int64_t d = 0; d < head_dim; ++d) {
                    ref[static_cast<size_t>(qh * head_dim + d)] +=
                        p * v_host[static_cast<size_t>(t * head_dim + d)];
                }
            }
        }
        const auto custom = read_fp16(custom_out);
        for (size_t i = 0; i < custom.size(); ++i) {
            if (std::fabs(custom[i] - ref[i]) > 5e-3f) {
                std::cerr << "attention custom mismatch at " << i << ": "
                          << custom[i] << " vs " << ref[i] << '\n';
                return 1;
            }
        }
    }

    auto step_hidden = make_tensor({1, 4}, {0.25f, -0.5f, 1.5f, 2.5f});
    hy_mt2::Tensor step_out({1, 4}, hy_mt2::DType::Float16); step_out.allocate();
    hy_mt2::decoder_layer_step(step_hidden, w, cos, sin, 2, 2, cfg, cache, step_out, ctx.stream());
    const auto step = read_fp16(step_out);
    const auto step_ref = read_fp16(step_hidden);
    for (size_t i = 0; i < step_ref.size(); ++i) {
        if (std::fabs(step[i] - step_ref[i]) > 5e-3f) {
            std::cerr << "decoder step residual mismatch at " << i << ": "
                      << step[i] << " vs " << step_ref[i] << '\n';
            return 1;
        }
    }

    std::cout << "[ok] decoder layer prefill/step synthetic residual path\n";
    return 0;
}
