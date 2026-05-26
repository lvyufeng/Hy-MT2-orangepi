#include "hy_mt2/language_model.h"

#include "hy_mt2/acl_context.h"
#include "hy_mt2/ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hy_mt2 {
namespace {

uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    const uint32_t mant = x & 0x7fffffu;
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

Tensor load_weight(WeightsIndex& index, const std::string& name) {
    return index.load_to_device_as(name, DType::Float16);
}

Tensor load_layer_weight(WeightsIndex& index, int64_t layer, const std::string& suffix) {
    return load_weight(index, "model.layers." + std::to_string(layer) + "." + suffix);
}

Tensor transpose_2d_device_to_device(const Tensor& src) {
    if (src.shape().size() != 2 || src.dtype() != DType::Float16) {
        throw std::runtime_error("transpose_2d_device_to_device expects a 2D fp16 tensor");
    }
    const int64_t rows = src.shape()[0];
    const int64_t cols = src.shape()[1];
    std::vector<uint16_t> host(static_cast<size_t>(rows) * cols);
    src.copy_to_host(host.data(), host.size() * sizeof(uint16_t));
    std::vector<uint16_t> transposed(static_cast<size_t>(rows) * cols);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            transposed[static_cast<size_t>(c) * rows + r] = host[static_cast<size_t>(r) * cols + c];
        }
    }
    Tensor dst({cols, rows}, DType::Float16);
    dst.copy_from_host(transposed.data(), transposed.size() * sizeof(uint16_t));
    return dst;
}

Tensor load_layer_weight_transposed(WeightsIndex& index, int64_t layer, const std::string& suffix) {
    return transpose_2d_device_to_device(load_layer_weight(index, layer, suffix));
}

void copy_tensor(const Tensor& src, Tensor& dst, aclrtStream stream) {
    if (src.shape() != dst.shape() || src.dtype() != dst.dtype()) throw std::runtime_error("copy_tensor shape/dtype mismatch");
    check_acl(aclrtMemcpyAsync(dst.data(), dst.size_bytes(), src.data(), src.size_bytes(),
                               ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
              "language_model copy_tensor");
    check_acl(aclrtSynchronizeStream(stream), "language_model copy_tensor sync");
}

void copy_row(const Tensor& src, int64_t src_row, Tensor& dst, int64_t dst_row, aclrtStream stream) {
    if (src.shape().size() != 2 || dst.shape().size() != 2 || src.shape()[1] != dst.shape()[1] || src.dtype() != dst.dtype()) {
        throw std::runtime_error("copy_row shape/dtype mismatch");
    }
    const size_t row_bytes = static_cast<size_t>(src.shape()[1]) * dtype_size(src.dtype());
    auto* s = static_cast<const uint8_t*>(src.data()) + static_cast<size_t>(src_row) * row_bytes;
    auto* d = static_cast<uint8_t*>(dst.data()) + static_cast<size_t>(dst_row) * row_bytes;
    check_acl(aclrtMemcpyAsync(d, row_bytes, s, row_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
              "language_model copy_row");
    check_acl(aclrtSynchronizeStream(stream), "language_model copy_row sync");
}

DecoderLayerWeights layer_weight_view(const LayerWeights& lw) {
    return DecoderLayerWeights{
        &lw.input_norm_w,
        &lw.post_norm_w,
        &lw.q_w,
        &lw.k_w,
        &lw.v_w,
        &lw.o_w,
        &lw.q_norm_w,
        &lw.k_norm_w,
        &lw.gate_w,
        &lw.up_w,
        &lw.down_w,
    };
}

DecoderLayerConfig decoder_config(const LanguageModelConfig& cfg) {
    return DecoderLayerConfig{cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rms_epsilon};
}

void check_config(const LanguageModelConfig& cfg) {
    if (cfg.hidden_size != cfg.num_q_heads * cfg.head_dim) throw std::runtime_error("LanguageModelConfig hidden/head mismatch");
    if (cfg.num_q_heads % cfg.num_kv_heads != 0) throw std::runtime_error("LanguageModelConfig GQA mismatch");
    if (cfg.num_layers <= 0 || cfg.vocab_size <= 0 || cfg.intermediate_size <= 0) throw std::runtime_error("LanguageModelConfig invalid size");
}

std::vector<LmHeadChunk> build_lm_head_chunks(const Tensor& embed, const LanguageModelConfig& cfg) {
    constexpr int64_t kChunkVocab = 8192;
    if (embed.shape() != std::vector<int64_t>{cfg.vocab_size, cfg.hidden_size}) {
        throw std::runtime_error("build_lm_head_chunks embedding shape mismatch");
    }
    std::vector<uint16_t> embed_host(static_cast<size_t>(cfg.vocab_size) * cfg.hidden_size);
    embed.copy_to_host(embed_host.data(), embed_host.size() * sizeof(uint16_t));

    std::vector<LmHeadChunk> chunks;
    chunks.reserve(static_cast<size_t>((cfg.vocab_size + kChunkVocab - 1) / kChunkVocab));
    std::vector<uint16_t> chunk_host(static_cast<size_t>(kChunkVocab) * cfg.hidden_size);
    for (int64_t start = 0; start < cfg.vocab_size; start += kChunkVocab) {
        const int64_t valid = std::min<int64_t>(kChunkVocab, cfg.vocab_size - start);
        std::fill(chunk_host.begin(), chunk_host.end(), uint16_t{0});
        for (int64_t v = 0; v < valid; ++v) {
            for (int64_t h = 0; h < cfg.hidden_size; ++h) {
                chunk_host[static_cast<size_t>(h) * kChunkVocab + v] =
                    embed_host[static_cast<size_t>(start + v) * cfg.hidden_size + h];
            }
        }
        LmHeadChunk chunk;
        chunk.start_vocab = start;
        chunk.valid_vocab = valid;
        chunk.weight = Tensor({cfg.hidden_size, kChunkVocab}, DType::Float16);
        chunk.weight.copy_from_host(chunk_host.data(), chunk_host.size() * sizeof(uint16_t));
        chunks.push_back(std::move(chunk));
    }
    return chunks;
}

}  // namespace

LanguageModelConfig default_hy_mt2_config() {
    return LanguageModelConfig{};
}

LanguageModelWeights load_language_model_weights(WeightsIndex& index,
                                                 const LanguageModelConfig& cfg) {
    check_config(cfg);
    LanguageModelWeights w;
    w.embed = load_weight(index, "model.embed_tokens.weight");
    w.final_norm_w = load_weight(index, "model.norm.weight");
    w.layers.resize(static_cast<size_t>(cfg.num_layers));
    for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
        auto& lw = w.layers[static_cast<size_t>(layer)];
        lw.input_norm_w = load_layer_weight(index, layer, "input_layernorm.weight");
        lw.post_norm_w = load_layer_weight(index, layer, "post_attention_layernorm.weight");
        lw.q_w = load_layer_weight_transposed(index, layer, "self_attn.q_proj.weight");
        lw.k_w = load_layer_weight_transposed(index, layer, "self_attn.k_proj.weight");
        lw.v_w = load_layer_weight_transposed(index, layer, "self_attn.v_proj.weight");
        lw.o_w = load_layer_weight_transposed(index, layer, "self_attn.o_proj.weight");
        lw.q_norm_w = load_layer_weight(index, layer, "self_attn.query_layernorm.weight");
        lw.k_norm_w = load_layer_weight(index, layer, "self_attn.key_layernorm.weight");
        lw.gate_w = load_layer_weight_transposed(index, layer, "mlp.gate_proj.weight");
        lw.up_w = load_layer_weight_transposed(index, layer, "mlp.up_proj.weight");
        lw.down_w = load_layer_weight_transposed(index, layer, "mlp.down_proj.weight");
    }
    w.lm_head_chunks = build_lm_head_chunks(w.embed, cfg);
    return w;
}

DecodeState make_decode_state(int64_t max_seq_len,
                              const LanguageModelConfig& cfg,
                              aclrtStream stream) {
    check_config(cfg);
    if (max_seq_len <= 0) throw std::runtime_error("make_decode_state max_seq_len must be positive");
    DecodeState state;
    state.max_seq_len = max_seq_len;
    state.seq_len = 0;
    const auto dcfg = decoder_config(cfg);
    state.layers.reserve(static_cast<size_t>(cfg.num_layers));
    for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
        state.layers.push_back(make_layer_cache(max_seq_len, dcfg, stream));
    }
    return state;
}

void build_rope_tables(int64_t max_seq_len,
                       const LanguageModelConfig& cfg,
                       Tensor& cos_table,
                       Tensor& sin_table) {
    check_config(cfg);
    if (max_seq_len <= 0) throw std::runtime_error("build_rope_tables max_seq_len must be positive");
    const int64_t half = cfg.head_dim / 2;
    double base = cfg.rope_theta;
    if (max_seq_len > cfg.max_position_embeddings && cfg.rope_scaling_factor > 0.0) {
        const double scaled = cfg.rope_scaling_factor * static_cast<double>(max_seq_len) /
                              static_cast<double>(cfg.max_position_embeddings) -
                              (cfg.rope_scaling_factor - 1.0);
        if (scaled > 1.0) base *= std::pow(scaled, static_cast<double>(cfg.head_dim) / static_cast<double>(cfg.head_dim - 2));
    }

    std::vector<uint16_t> cos_host(static_cast<size_t>(max_seq_len * half));
    std::vector<uint16_t> sin_host(static_cast<size_t>(max_seq_len * half));
    for (int64_t t = 0; t < max_seq_len; ++t) {
        for (int64_t i = 0; i < half; ++i) {
            const double inv = std::pow(base, -2.0 * static_cast<double>(i) / static_cast<double>(cfg.head_dim));
            const double theta = static_cast<double>(t) * inv;
            cos_host[static_cast<size_t>(t * half + i)] = f32_to_f16_bits(static_cast<float>(std::cos(theta)));
            sin_host[static_cast<size_t>(t * half + i)] = f32_to_f16_bits(static_cast<float>(std::sin(theta)));
        }
    }
    cos_table = Tensor({max_seq_len, half}, DType::Float16);
    sin_table = Tensor({max_seq_len, half}, DType::Float16);
    cos_table.copy_from_host(cos_host.data(), cos_host.size() * sizeof(uint16_t));
    sin_table.copy_from_host(sin_host.data(), sin_host.size() * sizeof(uint16_t));
}

Tensor prefill(const std::vector<int32_t>& token_ids,
               const LanguageModelWeights& weights,
               const LanguageModelConfig& cfg,
               const Tensor& cos_table,
               const Tensor& sin_table,
               DecodeState& state,
               aclrtStream stream) {
    check_config(cfg);
    const int64_t T = static_cast<int64_t>(token_ids.size());
    if (T <= 0) throw std::runtime_error("prefill token_ids must be non-empty");
    if (T > state.max_seq_len) throw std::runtime_error("prefill exceeds max_seq_len");
    if (state.seq_len != 0) throw std::runtime_error("prefill expects an empty decode state");
    if (cos_table.shape().size() != 2 || cos_table.shape()[0] < T || sin_table.shape() != cos_table.shape()) {
        throw std::runtime_error("prefill RoPE tables too short");
    }
    if (weights.layers.size() != static_cast<size_t>(cfg.num_layers) || state.layers.size() != weights.layers.size()) {
        throw std::runtime_error("prefill layer count mismatch");
    }

    Tensor hidden({T, cfg.hidden_size}, DType::Float16); hidden.allocate();
    Tensor next({T, cfg.hidden_size}, DType::Float16); next.allocate();
    embedding_lookup(weights.embed, token_ids, hidden, stream);

    std::vector<int32_t> row_to_t(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) row_to_t[static_cast<size_t>(t)] = static_cast<int32_t>(t);

    const auto dcfg = decoder_config(cfg);
    for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
        const auto& lw = weights.layers[static_cast<size_t>(layer)];
        decoder_layer_prefill(hidden, layer_weight_view(lw), cos_table, sin_table,
                              row_to_t, dcfg, state.layers[static_cast<size_t>(layer)], next, stream);
        copy_tensor(next, hidden, stream);
    }
    state.seq_len = T;

    Tensor last_hidden({1, cfg.hidden_size}, DType::Float16); last_hidden.allocate();
    copy_row(hidden, T - 1, last_hidden, 0, stream);
    return last_hidden;
}

int64_t lm_head_greedy(const Tensor& last_hidden_1xH,
                       const LanguageModelWeights& weights,
                       const LanguageModelConfig& cfg,
                       aclrtStream stream) {
    check_config(cfg);
    if (last_hidden_1xH.shape() != std::vector<int64_t>{1, cfg.hidden_size}) {
        throw std::runtime_error("lm_head_greedy hidden must be [1, hidden_size]");
    }
    if (weights.final_norm_w.shape() != std::vector<int64_t>{cfg.hidden_size}) {
        throw std::runtime_error("lm_head_greedy final_norm shape mismatch");
    }
    if (weights.lm_head_chunks.empty()) throw std::runtime_error("lm_head_greedy missing lm_head chunks");

    Tensor normed({1, cfg.hidden_size}, DType::Float16); normed.allocate();
    rms_norm(last_hidden_1xH, weights.final_norm_w, normed, cfg.rms_epsilon, stream);

    const int64_t chunk_vocab = weights.lm_head_chunks.front().weight.shape()[1];
    Tensor logits({1, chunk_vocab}, DType::Float16); logits.allocate();
    std::vector<uint16_t> logits_host(static_cast<size_t>(chunk_vocab));

    int64_t best = 0;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (const auto& chunk : weights.lm_head_chunks) {
        matmul_b_natural(normed, chunk.weight, logits, stream);
        logits.copy_to_host(logits_host.data(), logits_host.size() * sizeof(uint16_t));
        for (int64_t i = 0; i < chunk.valid_vocab; ++i) {
            const float v = f16_bits_to_f32(logits_host[static_cast<size_t>(i)]);
            if (v > best_logit) {
                best_logit = v;
                best = chunk.start_vocab + i;
            }
        }
    }
    return best;
}

int64_t decode_step_greedy(int32_t token_id,
                           const LanguageModelWeights& weights,
                           const LanguageModelConfig& cfg,
                           const Tensor& cos_table,
                           const Tensor& sin_table,
                           DecodeState& state,
                           aclrtStream stream) {
    check_config(cfg);
    if (state.seq_len >= state.max_seq_len) throw std::runtime_error("decode_step_greedy state full");
    if (weights.layers.size() != static_cast<size_t>(cfg.num_layers) || state.layers.size() != weights.layers.size()) {
        throw std::runtime_error("decode_step_greedy layer count mismatch");
    }

    Tensor hidden({1, cfg.hidden_size}, DType::Float16); hidden.allocate();
    Tensor next({1, cfg.hidden_size}, DType::Float16); next.allocate();
    embedding_lookup(weights.embed, {token_id}, hidden, stream);

    const auto dcfg = decoder_config(cfg);
    const int32_t pos = static_cast<int32_t>(state.seq_len);
    for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
        const auto& lw = weights.layers[static_cast<size_t>(layer)];
        decoder_layer_step(hidden, layer_weight_view(lw), cos_table, sin_table,
                           pos, state.seq_len, dcfg, state.layers[static_cast<size_t>(layer)], next, stream);
        copy_tensor(next, hidden, stream);
    }
    ++state.seq_len;
    return lm_head_greedy(hidden, weights, cfg, stream);
}

bool is_eos(int64_t token_id, const LanguageModelConfig& cfg) {
    return token_id == cfg.eos_token_id;
}

}  // namespace hy_mt2
