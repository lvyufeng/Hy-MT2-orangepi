#include "hy_mt2/acl_context.h"
#include "hy_mt2/ops.h"
#include "hy_mt2/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
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

void check_rope(hy_mt2::AclContext& ctx, int64_t N) {
    constexpr int64_t D = 128;
    constexpr int64_t Half = D / 2;
    std::vector<float> x_vals(static_cast<size_t>(N * D));
    std::vector<float> cos_vals(static_cast<size_t>(4 * Half));
    std::vector<float> sin_vals(static_cast<size_t>(4 * Half));
    for (size_t i = 0; i < x_vals.size(); ++i) x_vals[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.01f;
    for (size_t i = 0; i < cos_vals.size(); ++i) cos_vals[i] = 0.75f + static_cast<float>(i % 7) * 0.01f;
    for (size_t i = 0; i < sin_vals.size(); ++i) sin_vals[i] = -0.2f + static_cast<float>(i % 5) * 0.02f;
    std::vector<int32_t> row_to_t(static_cast<size_t>(N));
    for (int64_t n = 0; n < N; ++n) row_to_t[static_cast<size_t>(n)] = static_cast<int32_t>((n * 2) % 4);

    auto x = make_fp16_tensor({N, D}, x_vals);
    auto cos = make_fp16_tensor({4, Half}, cos_vals);
    auto sin = make_fp16_tensor({4, Half}, sin_vals);
    hy_mt2::Tensor out({N, D}, hy_mt2::DType::Float16); out.allocate();
    hy_mt2::apply_rope_full(x, cos, sin, row_to_t, out, ctx.stream());
    const auto got = read_fp16(out);
    const auto x_host = read_fp16(x);
    const auto cos_host = read_fp16(cos);
    const auto sin_host = read_fp16(sin);
    for (int64_t n = 0; n < N; ++n) {
        const int64_t pos = row_to_t[static_cast<size_t>(n)];
        for (int64_t i = 0; i < Half; ++i) {
            const float x1 = x_host[static_cast<size_t>(n * D + i)];
            const float x2 = x_host[static_cast<size_t>(n * D + Half + i)];
            const float c = cos_host[static_cast<size_t>(pos * Half + i)];
            const float s = sin_host[static_cast<size_t>(pos * Half + i)];
            const float y1 = x1 * c - x2 * s;
            const float y2 = x2 * c + x1 * s;
            if (!close(got[static_cast<size_t>(n * D + i)], y1, 5e-3f) ||
                !close(got[static_cast<size_t>(n * D + Half + i)], y2, 5e-3f)) {
                std::cerr << "rope mismatch N=" << N << " row=" << n << " i=" << i << '\n';
                std::cerr << got[static_cast<size_t>(n * D + i)] << " vs " << y1 << ", "
                          << got[static_cast<size_t>(n * D + Half + i)] << " vs " << y2 << '\n';
                std::exit(1);
            }
        }
    }
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

    check_rope(ctx, 1);
    check_rope(ctx, 4);
    check_rope(ctx, 16);

    if (hy_mt2::has_rms_norm128_custom()) {
        for (int64_t N : {1, 4, 16}) {
            constexpr int64_t D = 128;
            std::vector<float> x_vals(static_cast<size_t>(N * D));
            std::vector<float> gamma_vals(D);
            for (size_t i = 0; i < x_vals.size(); ++i) x_vals[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.02f;
            for (size_t i = 0; i < gamma_vals.size(); ++i) gamma_vals[i] = 0.5f + static_cast<float>(i % 5) * 0.05f;
            auto x = make_fp16_tensor({N, D}, x_vals);
            auto gamma_t = make_fp16_tensor({D}, gamma_vals);
            hy_mt2::Tensor out({N, D}, hy_mt2::DType::Float16); out.allocate();
            const double eps = 1e-5;
            hy_mt2::rms_norm128_custom(x, gamma_t, eps, out, ctx.stream());
            const auto got = read_fp16(out);
            const auto x_host = read_fp16(x);
            const auto g_host = read_fp16(gamma_t);
            for (int64_t n = 0; n < N; ++n) {
                double sumSq = 0.0;
                for (int64_t i = 0; i < D; ++i) {
                    const float v = x_host[static_cast<size_t>(n * D + i)];
                    sumSq += static_cast<double>(v) * v;
                }
                const float invRms = static_cast<float>(1.0 / std::sqrt(sumSq / D + eps));
                for (int64_t i = 0; i < D; ++i) {
                    const float ref = x_host[static_cast<size_t>(n * D + i)] * invRms * g_host[static_cast<size_t>(i)];
                    if (!close(got[static_cast<size_t>(n * D + i)], ref, 5e-3f)) {
                        std::cerr << "rms_norm128 mismatch N=" << N << " row=" << n << " i=" << i << ": "
                                  << got[static_cast<size_t>(n * D + i)] << " vs " << ref << '\n';
                        std::exit(1);
                    }
                }
            }
        }
    }

    if (hy_mt2::has_rms_norm2048_custom()) {
        for (int64_t N : {1, 4}) {
            constexpr int64_t D = 2048;
            std::vector<float> x_vals(static_cast<size_t>(N * D));
            std::vector<float> gamma_vals(D);
            for (size_t i = 0; i < x_vals.size(); ++i) x_vals[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.02f;
            for (size_t i = 0; i < gamma_vals.size(); ++i) gamma_vals[i] = 0.5f + static_cast<float>(i % 5) * 0.05f;
            auto x = make_fp16_tensor({N, D}, x_vals);
            auto gamma_t = make_fp16_tensor({D}, gamma_vals);
            hy_mt2::Tensor out({N, D}, hy_mt2::DType::Float16); out.allocate();
            const double eps = 1e-5;
            hy_mt2::rms_norm2048_custom(x, gamma_t, eps, out, ctx.stream());
            const auto got = read_fp16(out);
            const auto x_host = read_fp16(x);
            const auto g_host = read_fp16(gamma_t);
            for (int64_t n = 0; n < N; ++n) {
                double sumSq = 0.0;
                for (int64_t i = 0; i < D; ++i) {
                    const float v = x_host[static_cast<size_t>(n * D + i)];
                    sumSq += static_cast<double>(v) * v;
                }
                const float invRms = static_cast<float>(1.0 / std::sqrt(sumSq / D + eps));
                for (int64_t i = 0; i < D; ++i) {
                    const float ref = x_host[static_cast<size_t>(n * D + i)] * invRms * g_host[static_cast<size_t>(i)];
                    if (!close(got[static_cast<size_t>(n * D + i)], ref, 1e-2f)) {
                        std::cerr << "rms_norm2048 mismatch N=" << N << " row=" << n << " i=" << i << ": "
                                  << got[static_cast<size_t>(n * D + i)] << " vs " << ref << '\n';
                        std::exit(1);
                    }
                }
            }
        }
    }

    if (hy_mt2::has_silu_mul_custom()) {
        for (int64_t N : {1, 4, 16}) {
            constexpr int64_t M = 256;
            std::vector<float> gate_vals(static_cast<size_t>(N * M));
            std::vector<float> up_vals(static_cast<size_t>(N * M));
            for (size_t i = 0; i < gate_vals.size(); ++i) gate_vals[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.05f;
            for (size_t i = 0; i < up_vals.size(); ++i) up_vals[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.07f;
            auto gate = make_fp16_tensor({N, M}, gate_vals);
            auto up = make_fp16_tensor({N, M}, up_vals);
            hy_mt2::Tensor out({N, M}, hy_mt2::DType::Float16); out.allocate();
            hy_mt2::silu_mul_custom(gate, up, out, ctx.stream());
            const auto got = read_fp16(out);
            const auto g_host = read_fp16(gate);
            const auto u_host = read_fp16(up);
            for (int64_t i = 0; i < N * M; ++i) {
                const float g = g_host[static_cast<size_t>(i)];
                const float u = u_host[static_cast<size_t>(i)];
                const float silu = g / (1.0f + std::exp(-g));
                const float ref = silu * u;
                if (!close(got[static_cast<size_t>(i)], ref, 5e-3f)) {
                    std::cerr << "silu_mul_custom mismatch N=" << N << " i=" << i << ": "
                              << got[static_cast<size_t>(i)] << " vs " << ref << '\n';
                    std::exit(1);
                }
            }
        }
    }

    std::cout << "[ok] baseline aclnn ops wrappers run on NPU\n";
    return 0;
}
