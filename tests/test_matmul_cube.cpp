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
        bits = sign;
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
    constexpr int64_t M = 1;
    constexpr int64_t K = 64;
    constexpr int64_t N = 128;
    std::vector<float> a(static_cast<size_t>(M * K));
    std::vector<float> b(static_cast<size_t>(K * N));
    for (int64_t k = 0; k < K; ++k) a[static_cast<size_t>(k)] = (static_cast<float>((k % 7) - 3)) * 0.125f;
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t n = 0; n < N; ++n) {
            b[static_cast<size_t>(k * N + n)] = (static_cast<float>(((k * 17 + n * 5) % 13) - 6)) * 0.0625f;
        }
    }

    auto ta = make_tensor({M, K}, a);
    auto tb = make_tensor({K, N}, b);
    hy_mt2::Tensor out({M, N}, hy_mt2::DType::Float16); out.allocate();
    hy_mt2::matmul_b_natural(ta, tb, out, ctx.stream());
    const auto got = read_fp16(out);

    for (int64_t n = 0; n < N; ++n) {
        float ref = 0.0f;
        for (int64_t k = 0; k < K; ++k) ref += a[static_cast<size_t>(k)] * b[static_cast<size_t>(k * N + n)];
        if (std::fabs(got[static_cast<size_t>(n)] - ref) > 0.08f) {
            std::cerr << "matmul_b_natural mismatch at " << n << ": " << got[static_cast<size_t>(n)]
                      << " vs " << ref << '\n';
            return 1;
        }
    }
    std::cout << "[ok] matmul_b_natural cube path matches CPU reference\n";
    return 0;
}
