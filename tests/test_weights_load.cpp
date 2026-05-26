#include "hy_mt2/acl_context.h"
#include "hy_mt2/weights.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void write_u64_le(std::ofstream& out, uint64_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void write_synthetic_safetensors(const std::string& path) {
    const std::vector<uint16_t> bf16_values = {
        0x3f80,  // 1.0
        0xbf00,  // -0.5
        0x4020,  // 2.5
        0x4040,  // 3.0
    };
    const std::vector<int32_t> ids = {120000, 120006};

    const std::string header =
        R"({"__metadata__":{"format":"pt"},)"
        R"("model.embed_tokens.weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},)"
        R"("ids":{"dtype":"I32","shape":[2],"data_offsets":[8,16]}})";

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to create synthetic safetensors");
    write_u64_le(out, header.size());
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(bf16_values.data()),
              static_cast<std::streamsize>(bf16_values.size() * sizeof(uint16_t)));
    out.write(reinterpret_cast<const char*>(ids.data()),
              static_cast<std::streamsize>(ids.size() * sizeof(int32_t)));
}

}  // namespace

int main() {
    using hy_mt2::AclContext;
    using hy_mt2::DType;
    using hy_mt2::WeightsIndex;

    const std::string path = "/tmp/hy_mt2_synthetic.safetensors";
    write_synthetic_safetensors(path);

    AclContext ctx(0);
    WeightsIndex index(path);
    if (index.size() != 2) {
        std::cerr << "unexpected tensor count: " << index.size() << '\n';
        return 1;
    }
    if (!index.contains("model.embed_tokens.weight") || !index.contains("ids")) {
        std::cerr << "missing expected tensor names\n";
        return 1;
    }

    const auto& embed_info = index.at("model.embed_tokens.weight");
    if (embed_info.dtype != DType::BFloat16 || embed_info.shape != std::vector<int64_t>({2, 2})) {
        std::cerr << "bad embed metadata\n";
        return 1;
    }

    auto embed = index.load_to_device_as("model.embed_tokens.weight", DType::Float16);
    std::vector<uint16_t> out(4);
    embed.copy_to_host(out.data(), out.size() * sizeof(uint16_t));
    const std::vector<uint16_t> expected = {0x3c00, 0xb800, 0x4100, 0x4200};
    if (out != expected) {
        std::cerr << "BF16->FP16 conversion mismatch\n";
        for (auto v : out) std::cerr << std::hex << v << ' ';
        std::cerr << '\n';
        return 1;
    }

    auto ids = index.load_to_device("ids");
    std::vector<int32_t> ids_out(2);
    ids.copy_to_host(ids_out.data(), ids_out.size() * sizeof(int32_t));
    if (ids_out != std::vector<int32_t>({120000, 120006})) {
        std::cerr << "ids roundtrip mismatch\n";
        return 1;
    }

    std::cout << "[ok] WeightsIndex parses metadata and loads BF16->FP16 tensors\n";
    return 0;
}
