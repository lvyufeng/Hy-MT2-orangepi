#include "hy_mt2/acl_context.h"
#include "hy_mt2/ops.h"
#include "hy_mt2/tensor.h"

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

struct Shape {
    int64_t m;
    int64_t n;
    int64_t k;
};

}  // namespace

int main(int argc, char** argv) {
    int iters = 20;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: " << argv[0] << " [--iters N]\n";
            return 0;
        }
    }

    const std::vector<Shape> shapes = {
        {1, 2048, 2048},
        {1, 512, 2048},
        {1, 6144, 2048},
        {1, 2048, 6144},
        {1, 120818, 2048},
        {16, 2048, 2048},
    };

    hy_mt2::AclContext ctx(0);
    std::cout << "path,M,N,K,iters,ms_per_iter,gflops\n";
    for (const auto& s : shapes) {
        auto a = filled({s.m, s.k}, 0.001f);
        auto b_transposed = filled({s.n, s.k}, 0.002f);
        auto b_natural = filled({s.k, s.n}, 0.002f);
        hy_mt2::Tensor out({s.m, s.n}, hy_mt2::DType::Float16); out.allocate();
        const double flops = 2.0 * static_cast<double>(s.m) * s.n * s.k;

        hy_mt2::matmul_b_transposed(a, b_transposed, out, ctx.stream());
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) hy_mt2::matmul_b_transposed(a, b_transposed, out, ctx.stream());
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
        std::cout << "aclnn_bt," << s.m << ',' << s.n << ',' << s.k << ',' << iters << ',' << ms << ',' << (flops / (ms * 1.0e6)) << '\n';

        hy_mt2::matmul_b_natural(a, b_natural, out, ctx.stream());
        start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) hy_mt2::matmul_b_natural(a, b_natural, out, ctx.stream());
        end = std::chrono::steady_clock::now();
        ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
        std::cout << "natural_or_cube," << s.m << ',' << s.n << ',' << s.k << ',' << iters << ',' << ms << ',' << (flops / (ms * 1.0e6)) << '\n';
    }
    return 0;
}
