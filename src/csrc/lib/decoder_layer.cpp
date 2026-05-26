#include "hy_mt2/decoder_layer.h"

#include "hy_mt2/acl_context.h"
#include "hy_mt2/ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

void check_ptr(const Tensor* t, const char* name) {
    if (t == nullptr) throw std::runtime_error(std::string("missing decoder layer weight: ") + name);
}

void copy_col_block(const Tensor& src, int64_t col_offset, Tensor& dst, aclrtStream stream) {
    const int64_t rows = src.shape()[0];
    const int64_t src_cols = src.shape()[1];
    const int64_t dst_cols = dst.shape()[1];
    const size_t elem = dtype_size(src.dtype());
    const size_t src_row_bytes = static_cast<size_t>(src_cols) * elem;
    const size_t dst_row_bytes = static_cast<size_t>(dst_cols) * elem;
    const size_t block_bytes = static_cast<size_t>(dst_cols) * elem;
    auto* s = static_cast<const uint8_t*>(src.data());
    auto* d = static_cast<uint8_t*>(dst.data());
    for (int64_t r = 0; r < rows; ++r) {
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(r) * dst_row_bytes, block_bytes,
                                   s + static_cast<size_t>(r) * src_row_bytes + static_cast<size_t>(col_offset) * elem,
                                   block_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_col_block");
    }
    check_acl(aclrtSynchronizeStream(stream), "copy_col_block sync");
}

void copy_head_to_seq(const Tensor& src_heads, int64_t head, int64_t heads_per_token,
                      Tensor& dst_seq, aclrtStream stream) {
    const int64_t tokens = dst_seq.shape()[0];
    const int64_t dim = dst_seq.shape()[1];
    const size_t row_bytes = static_cast<size_t>(dim) * dtype_size(src_heads.dtype());
    auto* s = static_cast<const uint8_t*>(src_heads.data());
    auto* d = static_cast<uint8_t*>(dst_seq.data());
    for (int64_t t = 0; t < tokens; ++t) {
        const int64_t src_row = t * heads_per_token + head;
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(t) * row_bytes, row_bytes,
                                   s + static_cast<size_t>(src_row) * row_bytes, row_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_head_to_seq");
    }
    check_acl(aclrtSynchronizeStream(stream), "copy_head_to_seq sync");
}

void copy_seq_to_head_block(const Tensor& src_seq, Tensor& dst, int64_t col_offset,
                            aclrtStream stream) {
    const int64_t rows = src_seq.shape()[0];
    const int64_t dst_cols = dst.shape()[1];
    const int64_t src_cols = src_seq.shape()[1];
    const size_t elem = dtype_size(src_seq.dtype());
    const size_t src_row_bytes = static_cast<size_t>(src_cols) * elem;
    const size_t dst_row_bytes = static_cast<size_t>(dst_cols) * elem;
    auto* s = static_cast<const uint8_t*>(src_seq.data());
    auto* d = static_cast<uint8_t*>(dst.data());
    for (int64_t r = 0; r < rows; ++r) {
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(r) * dst_row_bytes + static_cast<size_t>(col_offset) * elem,
                                   src_row_bytes,
                                   s + static_cast<size_t>(r) * src_row_bytes,
                                   src_row_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_seq_to_head_block");
    }
    check_acl(aclrtSynchronizeStream(stream), "copy_seq_to_head_block sync");
}

void copy_heads_from_cols(const Tensor& src, int64_t heads, int64_t head_dim,
                          Tensor& dst, aclrtStream stream) {
    const int64_t rows = src.shape()[0];
    const int64_t src_cols = src.shape()[1];
    const size_t elem = dtype_size(src.dtype());
    const size_t src_row_bytes = static_cast<size_t>(src_cols) * elem;
    const size_t head_bytes = static_cast<size_t>(head_dim) * elem;
    auto* s = static_cast<const uint8_t*>(src.data());
    auto* d = static_cast<uint8_t*>(dst.data());
    for (int64_t t = 0; t < rows; ++t) {
        for (int64_t h = 0; h < heads; ++h) {
            check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(t * heads + h) * head_bytes, head_bytes,
                                       s + static_cast<size_t>(t) * src_row_bytes + static_cast<size_t>(h * head_dim) * elem,
                                       head_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                      "copy_heads_from_cols");
        }
    }
    check_acl(aclrtSynchronizeStream(stream), "copy_heads_from_cols sync");
}

void copy_cache_head_to_seq(const Tensor& cache, int64_t head,
                            int64_t head_dim, int64_t rows, Tensor& dst_seq,
                            aclrtStream stream) {
    const size_t elem = dtype_size(cache.dtype());
    const size_t cache_row_bytes = static_cast<size_t>(cache.shape()[1]) * elem;
    const size_t head_bytes = static_cast<size_t>(head_dim) * elem;
    auto* s = static_cast<const uint8_t*>(cache.data());
    auto* d = static_cast<uint8_t*>(dst_seq.data());
    for (int64_t r = 0; r < rows; ++r) {
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(r) * head_bytes, head_bytes,
                                   s + static_cast<size_t>(r) * cache_row_bytes + static_cast<size_t>(head * head_dim) * elem,
                                   head_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_cache_head_to_seq");
    }
    check_acl(aclrtSynchronizeStream(stream), "copy_cache_head_to_seq sync");
}

bool matmul_weight_shape_ok(const Tensor* t, int64_t out_dim, int64_t in_dim) {
    return t->shape() == std::vector<int64_t>{out_dim, in_dim};
}

int64_t infer_intermediate(const Tensor& gate_proj_weight, int64_t hidden) {
    const auto& s = gate_proj_weight.shape();
    if (s.size() != 2 || s[1] != hidden) throw std::runtime_error("decoder gate_proj shape mismatch");
    return s[0];
}

void validate_shapes(const Tensor& hidden,
                     const DecoderLayerWeights& w,
                     const DecoderLayerConfig& c,
                     const Tensor& out) {
    check_ptr(w.input_norm_weight, "input_norm_weight");
    check_ptr(w.post_attention_norm_weight, "post_attention_norm_weight");
    check_ptr(w.q_proj_weight, "q_proj_weight");
    check_ptr(w.k_proj_weight, "k_proj_weight");
    check_ptr(w.v_proj_weight, "v_proj_weight");
    check_ptr(w.o_proj_weight, "o_proj_weight");
    check_ptr(w.q_norm_weight, "q_norm_weight");
    check_ptr(w.k_norm_weight, "k_norm_weight");
    check_ptr(w.gate_proj_weight, "gate_proj_weight");
    check_ptr(w.up_proj_weight, "up_proj_weight");
    check_ptr(w.down_proj_weight, "down_proj_weight");

    if (hidden.shape().size() != 2 || out.shape() != hidden.shape()) {
        throw std::runtime_error("decoder layer hidden/out must be [T, H] and same shape");
    }
    if (hidden.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("decoder layer requires fp16 hidden/out");
    }
    if (c.num_q_heads <= 0 || c.num_kv_heads <= 0 || c.head_dim <= 0 ||
        c.num_q_heads % c.num_kv_heads != 0) {
        throw std::runtime_error("decoder layer invalid attention config");
    }

    const int64_t hidden_size = hidden.shape()[1];
    const int64_t q_dim = c.num_q_heads * c.head_dim;
    const int64_t kv_dim = c.num_kv_heads * c.head_dim;
    const int64_t intermediate = infer_intermediate(*w.gate_proj_weight, hidden_size);
    if (w.input_norm_weight->shape() != std::vector<int64_t>{hidden_size} ||
        w.post_attention_norm_weight->shape() != std::vector<int64_t>{hidden_size} ||
        !matmul_weight_shape_ok(w.q_proj_weight, q_dim, hidden_size) ||
        !matmul_weight_shape_ok(w.k_proj_weight, kv_dim, hidden_size) ||
        !matmul_weight_shape_ok(w.v_proj_weight, kv_dim, hidden_size) ||
        !matmul_weight_shape_ok(w.o_proj_weight, hidden_size, q_dim) ||
        w.q_norm_weight->shape() != std::vector<int64_t>{c.head_dim} ||
        w.k_norm_weight->shape() != std::vector<int64_t>{c.head_dim} ||
        !matmul_weight_shape_ok(w.up_proj_weight, intermediate, hidden_size) ||
        !matmul_weight_shape_ok(w.down_proj_weight, hidden_size, intermediate)) {
        throw std::runtime_error("decoder layer weight shape mismatch");
    }
}

void write_kv_cache(const Tensor& k_rope,
                    const Tensor& v_full,
                    int64_t num_kv_heads,
                    int64_t head_dim,
                    int64_t cache_offset,
                    LayerCache& cache,
                    aclrtStream stream) {
    const int64_t T = v_full.shape()[0];
    const int64_t kv_dim = num_kv_heads * head_dim;
    const size_t elem = dtype_size(DType::Float16);
    const size_t head_bytes = static_cast<size_t>(head_dim) * elem;
    const size_t row_bytes = static_cast<size_t>(kv_dim) * elem;
    auto* ks = static_cast<const uint8_t*>(k_rope.data());
    auto* kd = static_cast<uint8_t*>(cache.k_cache.data());
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < num_kv_heads; ++h) {
            check_acl(aclrtMemcpyAsync(kd + static_cast<size_t>(cache_offset + t) * row_bytes + static_cast<size_t>(h) * head_bytes,
                                       head_bytes,
                                       ks + static_cast<size_t>(t * num_kv_heads + h) * head_bytes,
                                       head_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                      "write k cache");
        }
    }

    auto* vs = static_cast<const uint8_t*>(v_full.data());
    auto* vd = static_cast<uint8_t*>(cache.v_cache.data());
    for (int64_t t = 0; t < T; ++t) {
        check_acl(aclrtMemcpyAsync(vd + static_cast<size_t>(cache_offset + t) * row_bytes,
                                   row_bytes,
                                   vs + static_cast<size_t>(t) * row_bytes,
                                   row_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "write v cache");
    }
    check_acl(aclrtSynchronizeStream(stream), "write kv cache sync");
}

void run_prefill_attention(const Tensor& q_rope,
                           const Tensor& k_rope,
                           const Tensor& v_full,
                           const std::vector<int32_t>& row_to_t,
                           const DecoderLayerConfig& c,
                           Tensor& attn_out,
                           aclrtStream stream) {
    const int64_t T = v_full.shape()[0];
    const int64_t q_per_kv = c.num_q_heads / c.num_kv_heads;
    Tensor causal_mask({T, T}, DType::Float16);
    std::vector<uint16_t> mask(static_cast<size_t>(T * T));
    for (int64_t r = 0; r < T; ++r) {
        for (int64_t col = 0; col < T; ++col) {
            mask[static_cast<size_t>(r) * T + col] = f32_to_f16_bits(row_to_t[col] <= row_to_t[r] ? 0.0f : -65504.0f);
        }
    }
    causal_mask.copy_from_host(mask.data(), mask.size() * sizeof(uint16_t));

    Tensor q_seq({T, c.head_dim}, DType::Float16); q_seq.allocate();
    Tensor k_seq({T, c.head_dim}, DType::Float16); k_seq.allocate();
    Tensor v_seq({T, c.head_dim}, DType::Float16); v_seq.allocate();
    Tensor scores({T, T}, DType::Float16); scores.allocate();
    Tensor scaled_scores({T, T}, DType::Float16); scaled_scores.allocate();
    Tensor masked_scores({T, T}, DType::Float16); masked_scores.allocate();
    Tensor probs({T, T}, DType::Float16); probs.allocate();
    Tensor ctx_seq({T, c.head_dim}, DType::Float16); ctx_seq.allocate();
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));

    for (int64_t qh = 0; qh < c.num_q_heads; ++qh) {
        const int64_t kvh = qh / q_per_kv;
        copy_head_to_seq(q_rope, qh, c.num_q_heads, q_seq, stream);
        copy_head_to_seq(k_rope, kvh, c.num_kv_heads, k_seq, stream);
        copy_col_block(v_full, kvh * c.head_dim, v_seq, stream);
        matmul_b_transposed(q_seq, k_seq, scores, stream);
        muls(scores, scale, scaled_scores, stream);
        add(scaled_scores, causal_mask, masked_scores, stream);
        softmax_last_dim(masked_scores, probs, stream);
        matmul(probs, v_seq, ctx_seq, stream);
        copy_seq_to_head_block(ctx_seq, attn_out, qh * c.head_dim, stream);
    }
}

void run_step_attention(const Tensor& q_rope,
                        const LayerCache& cache,
                        int64_t context,
                        const DecoderLayerConfig& c,
                        Tensor& attn_out,
                        aclrtStream stream) {
    const int64_t q_per_kv = c.num_q_heads / c.num_kv_heads;
    Tensor q_seq({1, c.head_dim}, DType::Float16); q_seq.allocate();
    Tensor k_seq({context, c.head_dim}, DType::Float16); k_seq.allocate();
    Tensor v_seq({context, c.head_dim}, DType::Float16); v_seq.allocate();
    Tensor scores({1, context}, DType::Float16); scores.allocate();
    Tensor scaled_scores({1, context}, DType::Float16); scaled_scores.allocate();
    Tensor probs({1, context}, DType::Float16); probs.allocate();
    Tensor ctx_seq({1, c.head_dim}, DType::Float16); ctx_seq.allocate();
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));

    for (int64_t qh = 0; qh < c.num_q_heads; ++qh) {
        const int64_t kvh = qh / q_per_kv;
        copy_head_to_seq(q_rope, qh, c.num_q_heads, q_seq, stream);
        copy_cache_head_to_seq(cache.k_cache, kvh, c.head_dim, context, k_seq, stream);
        copy_cache_head_to_seq(cache.v_cache, kvh, c.head_dim, context, v_seq, stream);
        matmul_b_transposed(q_seq, k_seq, scores, stream);
        muls(scores, scale, scaled_scores, stream);
        softmax_last_dim(scaled_scores, probs, stream);
        matmul(probs, v_seq, ctx_seq, stream);
        copy_seq_to_head_block(ctx_seq, attn_out, qh * c.head_dim, stream);
    }
}

void run_layer_core(const Tensor& hidden,
                    const DecoderLayerWeights& weights,
                    const Tensor& cos_table,
                    const Tensor& sin_table,
                    const std::vector<int32_t>& row_to_t,
                    const DecoderLayerConfig& config,
                    LayerCache& cache,
                    int64_t cache_offset,
                    Tensor& out,
                    aclrtStream stream) {
    validate_shapes(hidden, weights, config, out);
    const int64_t T = hidden.shape()[0];
    const int64_t hidden_size = hidden.shape()[1];
    const int64_t q_dim = config.num_q_heads * config.head_dim;
    const int64_t kv_dim = config.num_kv_heads * config.head_dim;
    const int64_t intermediate = weights.gate_proj_weight->shape()[0];
    if (static_cast<int64_t>(row_to_t.size()) != T) throw std::runtime_error("decoder row_to_t size mismatch");
    if (cache.k_cache.shape()[1] != kv_dim || cache.v_cache.shape() != cache.k_cache.shape() ||
        cache_offset + T > cache.k_cache.shape()[0]) {
        throw std::runtime_error("decoder cache shape/offset mismatch");
    }

    Tensor normed({T, hidden_size}, DType::Float16); normed.allocate();
    rms_norm(hidden, *weights.input_norm_weight, normed, config.rms_epsilon, stream);

    Tensor q_full({T, q_dim}, DType::Float16); q_full.allocate();
    Tensor k_full({T, kv_dim}, DType::Float16); k_full.allocate();
    Tensor v_full({T, kv_dim}, DType::Float16); v_full.allocate();
    matmul_b_transposed(normed, *weights.q_proj_weight, q_full, stream);
    matmul_b_transposed(normed, *weights.k_proj_weight, k_full, stream);
    matmul_b_transposed(normed, *weights.v_proj_weight, v_full, stream);

    Tensor q_heads({T * config.num_q_heads, config.head_dim}, DType::Float16); q_heads.allocate();
    Tensor k_heads({T * config.num_kv_heads, config.head_dim}, DType::Float16); k_heads.allocate();
    copy_heads_from_cols(q_full, config.num_q_heads, config.head_dim, q_heads, stream);
    copy_heads_from_cols(k_full, config.num_kv_heads, config.head_dim, k_heads, stream);

    Tensor q_normed({T * config.num_q_heads, config.head_dim}, DType::Float16); q_normed.allocate();
    Tensor k_normed({T * config.num_kv_heads, config.head_dim}, DType::Float16); k_normed.allocate();
    rms_norm(q_heads, *weights.q_norm_weight, q_normed, config.rms_epsilon, stream);
    rms_norm(k_heads, *weights.k_norm_weight, k_normed, config.rms_epsilon, stream);

    std::vector<int32_t> q_row_to_t(static_cast<size_t>(T * config.num_q_heads));
    std::vector<int32_t> k_row_to_t(static_cast<size_t>(T * config.num_kv_heads));
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < config.num_q_heads; ++h) q_row_to_t[static_cast<size_t>(t * config.num_q_heads + h)] = row_to_t[static_cast<size_t>(t)];
        for (int64_t h = 0; h < config.num_kv_heads; ++h) k_row_to_t[static_cast<size_t>(t * config.num_kv_heads + h)] = row_to_t[static_cast<size_t>(t)];
    }

    Tensor q_rope({T * config.num_q_heads, config.head_dim}, DType::Float16); q_rope.allocate();
    Tensor k_rope({T * config.num_kv_heads, config.head_dim}, DType::Float16); k_rope.allocate();
    apply_rope_full(q_normed, cos_table, sin_table, q_row_to_t, q_rope, stream);
    apply_rope_full(k_normed, cos_table, sin_table, k_row_to_t, k_rope, stream);

    write_kv_cache(k_rope, v_full, config.num_kv_heads, config.head_dim, cache_offset, cache, stream);

    Tensor attn_out({T, q_dim}, DType::Float16); attn_out.allocate();
    if (T == 1) {
        run_step_attention(q_rope, cache, cache_offset + 1, config, attn_out, stream);
    } else {
        run_prefill_attention(q_rope, k_rope, v_full, row_to_t, config, attn_out, stream);
    }

    Tensor attn_proj({T, hidden_size}, DType::Float16); attn_proj.allocate();
    matmul_b_transposed(attn_out, *weights.o_proj_weight, attn_proj, stream);

    Tensor after_attn({T, hidden_size}, DType::Float16); after_attn.allocate();
    add(hidden, attn_proj, after_attn, stream);

    Tensor mlp_in({T, hidden_size}, DType::Float16); mlp_in.allocate();
    rms_norm(after_attn, *weights.post_attention_norm_weight, mlp_in, config.rms_epsilon, stream);

    Tensor gate({T, intermediate}, DType::Float16); gate.allocate();
    Tensor up({T, intermediate}, DType::Float16); up.allocate();
    Tensor gate_act({T, intermediate}, DType::Float16); gate_act.allocate();
    Tensor gated({T, intermediate}, DType::Float16); gated.allocate();
    Tensor mlp_out({T, hidden_size}, DType::Float16); mlp_out.allocate();
    matmul_b_transposed(mlp_in, *weights.gate_proj_weight, gate, stream);
    matmul_b_transposed(mlp_in, *weights.up_proj_weight, up, stream);
    silu(gate, gate_act, stream);
    mul(gate_act, up, gated, stream);
    matmul_b_transposed(gated, *weights.down_proj_weight, mlp_out, stream);
    add(after_attn, mlp_out, out, stream);
}

}  // namespace

void decoder_layer_prefill(const Tensor& hidden,
                           const DecoderLayerWeights& weights,
                           const Tensor& cos_table,
                           const Tensor& sin_table,
                           const std::vector<int32_t>& row_to_t,
                           const DecoderLayerConfig& config,
                           LayerCache& cache,
                           Tensor& out,
                           aclrtStream stream) {
    run_layer_core(hidden, weights, cos_table, sin_table, row_to_t, config, cache, 0, out, stream);
}

void decoder_layer_step(const Tensor& hidden,
                        const DecoderLayerWeights& weights,
                        const Tensor& cos_table,
                        const Tensor& sin_table,
                        int32_t pos,
                        int64_t cache_len,
                        const DecoderLayerConfig& config,
                        LayerCache& cache,
                        Tensor& out,
                        aclrtStream stream) {
    if (hidden.shape().empty() || hidden.shape()[0] != 1) {
        throw std::runtime_error("decoder_layer_step hidden must be [1, H]");
    }
    std::vector<int32_t> row_to_t{pos};
    run_layer_core(hidden, weights, cos_table, sin_table, row_to_t, config, cache, cache_len, out, stream);
}

LayerCache make_layer_cache(int64_t max_seq_len,
                            const DecoderLayerConfig& config,
                            aclrtStream stream) {
    if (max_seq_len <= 0) throw std::runtime_error("layer cache max_seq_len must be positive");
    LayerCache cache;
    const int64_t kv_dim = config.num_kv_heads * config.head_dim;
    cache.k_cache = Tensor({max_seq_len, kv_dim}, DType::Float16);
    cache.v_cache = Tensor({max_seq_len, kv_dim}, DType::Float16);
    cache.k_cache.allocate();
    cache.v_cache.allocate();
    check_acl(aclrtMemsetAsync(cache.k_cache.data(), cache.k_cache.size_bytes(), 0,
                               cache.k_cache.size_bytes(), stream), "memset k cache");
    check_acl(aclrtMemsetAsync(cache.v_cache.data(), cache.v_cache.size_bytes(), 0,
                               cache.v_cache.size_bytes(), stream), "memset v cache");
    check_acl(aclrtSynchronizeStream(stream), "make_layer_cache sync");
    return cache;
}

}  // namespace hy_mt2
