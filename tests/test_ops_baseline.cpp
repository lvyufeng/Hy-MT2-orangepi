#include "hy_mt2/acl_context.h"
#include "hy_mt2/ops.h"
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

hy_mt2::Tensor make_fp16_tensor(const std::vector<int64_t>& shape, const std::vector<float>& values) {
    std::vector<uint16_t> bits(values.size());
    for (size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_f16_bits(values[i]);
    hy_mt2::Tensor t(shape, hy_mt2::DType::Float16);
    t.allocate();
    t.copy_from_host(bits.data(), bits.size() * sizeof(uint16_t));
    return t;
}

std::vector<float> read_fp16(const hy_mt2::Tensor& t) {
    std::vector<uint16_t> bits(t.numel());
    t.copy_to_host(bits.data(), bits.size() * sizeof(uint16_t));
    std::vector<float> out(bits.size());
    for (size_t i = 0; i < bits.size(); ++i) out[i] = f16_bits_to_f32(bits[i]);
    return out;
}

bool close(float a, float b, float tol = 5e-3f) {
    return std::fabs(a - b) <= tol;
}

}  // namespace

int main() {
    hy_mt2::AclContext ctx(0);

    auto a = make_fp16_tensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto b = make_fp16_tensor({2, 2}, {0.5f, 1.0f, 1.5f, 2.0f});

    hy_mt2::Tensor sum({2, 2}, hy_mt2::DType::Float16); sum.allocate();
    hy_mt2::add(a, b, sum, ctx.stream());
    const auto sum_out = read_fp16(sum);
    const std::vector<float> sum_ref = {1.5f, 3.0f, 4.5f, 6.0f};
    for (size_t i = 0; i < sum_ref.size(); ++i) {
        if (!close(sum_out[i], sum_ref[i])) {
            std::cerr << "add mismatch at " << i << ": " << sum_out[i] << " vs " << sum_ref[i] << '\n';
            return 1;
        }
    }

    hy_mt2::Tensor prod({2, 2}, hy_mt2::DType::Float16); prod.allocate();
    hy_mt2::mul(a, b, prod, ctx.stream());
    const auto prod_out = read_fp16(prod);
    const std::vector<float> prod_ref = {0.5f, 2.0f, 4.5f, 8.0f};
    for (size_t i = 0; i < prod_ref.size(); ++i) {
        if (!close(prod_out[i], prod_ref[i])) {
            std::cerr << "mul mismatch\n";
            return 1;
        }
    }

    hy_mt2::Tensor mm({2, 2}, hy_mt2::DType::Float16); mm.allocate();
    hy_mt2::matmul(a, b, mm, ctx.stream());
    const auto mm_out = read_fp16(mm);
    const std::vector<float> mm_ref = {3.5f, 5.0f, 7.5f, 11.0f};
    for (size_t i = 0; i < mm_ref.size(); ++i) {
        if (!close(mm_out[i], mm_ref[i], 2e-2f)) {
            std::cerr << "matmul mismatch at " << i << ": " << mm_out[i] << " vs " << mm_ref[i] << '\n';
            return 1;
        }
    }

    hy_mt2::Tensor sm({2, 2}, hy_mt2::DType::Float16); sm.allocate();
    hy_mt2::softmax_last_dim(a, sm, ctx.stream());
    const auto sm_out = read_fp16(sm);
    const float e1 = std::exp(1.0f), e2 = std::exp(2.0f);
    if (!close(sm_out[0], e1 / (e1 + e2), 2e-3f) || !close(sm_out[1], e2 / (e1 + e2), 2e-3f)) {
        std::cerr << "softmax mismatch\n";
        return 1;
    }

    auto gamma = make_fp16_tensor({2}, {1.0f, 1.0f});
    hy_mt2::Tensor norm({2, 2}, hy_mt2::DType::Float16); norm.allocate();
    hy_mt2::rms_norm(a, gamma, norm, 1e-5, ctx.stream());
    const auto norm_out = read_fp16(norm);
    if (norm_out.empty()) return 1;

    std::cout << "[ok] baseline aclnn ops wrappers run on NPU\n";
    return 0;
}
