#include "hy_mt2/acl_context.h"
#include "hy_mt2/decoder_layer.h"
#include "hy_mt2/tensor.h"

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
