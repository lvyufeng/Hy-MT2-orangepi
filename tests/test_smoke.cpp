// Phase 0 smoke test: stand up an AclContext, allocate a small fp16
// Tensor, copy round-trip a few values to/from device. Validates the
// CANN install + the engine library link before any model code lands.

#include "hy_mt2/acl_context.h"
#include "hy_mt2/tensor.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

uint16_t f32_to_f16_bits(float v) {
    // Quick non-IEEE-exact host helper for the smoke test. We only need
    // bit-equality across a roundtrip, not numeric correctness.
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = bits & 0x7fffff;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00);
    return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
}

}  // namespace

int main() {
    using hy_mt2::AclContext;
    using hy_mt2::DType;
    using hy_mt2::Tensor;

    AclContext ctx(/*device_id=*/0);
    std::cout << "[ok] AclContext on device " << ctx.device_id() << '\n';

    const std::vector<float> host_in = {1.0f, -0.5f, 2.5f, 3.0f};
    std::vector<uint16_t> host_in_f16(host_in.size());
    for (size_t i = 0; i < host_in.size(); ++i) {
        host_in_f16[i] = f32_to_f16_bits(host_in[i]);
    }

    Tensor t({static_cast<int64_t>(host_in.size())}, DType::Float16);
    t.allocate();
    t.copy_from_host(host_in_f16.data(), host_in_f16.size() * sizeof(uint16_t));

    std::vector<uint16_t> host_out(host_in.size(), 0);
    t.copy_to_host(host_out.data(), host_out.size() * sizeof(uint16_t));

    for (size_t i = 0; i < host_in.size(); ++i) {
        if (host_out[i] != host_in_f16[i]) {
            std::cerr << "round-trip mismatch at " << i << ": "
                      << std::hex << host_out[i] << " vs " << host_in_f16[i] << '\n';
            return 1;
        }
    }
    std::cout << "[ok] Tensor H2D/D2H round-trip " << host_in.size() << " fp16 values\n";
    return 0;
}
