#pragma once

#include "hy_mt2/tensor.h"

#include <acl/acl.h>

#include <cstdint>
#include <vector>

namespace hy_mt2 {

struct DecoderLayerConfig {
    int64_t num_q_heads;     // 16
    int64_t num_kv_heads;    // 4
    int64_t head_dim;        // 128
    double rms_epsilon;      // 1e-5
};

struct DecoderLayerWeights {
    const Tensor* input_norm_weight;
    const Tensor* post_attention_norm_weight;
    const Tensor* q_proj_weight;          // [num_q_heads * head_dim, hidden]
    const Tensor* k_proj_weight;          // [num_kv_heads * head_dim, hidden]
    const Tensor* v_proj_weight;          // [num_kv_heads * head_dim, hidden]
    const Tensor* o_proj_weight;          // [hidden, num_q_heads * head_dim]
    const Tensor* q_norm_weight;          // [head_dim]
    const Tensor* k_norm_weight;          // [head_dim]
    const Tensor* gate_proj_weight;       // [intermediate, hidden]
    const Tensor* up_proj_weight;         // [intermediate, hidden]
    const Tensor* down_proj_weight;       // [hidden, intermediate]
};

struct LayerCache {
    Tensor k_cache;  // [max_seq, num_kv_heads * head_dim] fp16, post-RoPE
    Tensor v_cache;  // [max_seq, num_kv_heads * head_dim] fp16
};

// Multi-token prefill. Runs the layer on hidden [T, H], writes K/V rows
// [0, T) into the layer's cache, returns the post-MLP residual stream in out.
void decoder_layer_prefill(const Tensor& hidden,
                           const DecoderLayerWeights& weights,
                           const Tensor& cos_table,
                           const Tensor& sin_table,
                           const std::vector<int32_t>& row_to_t,
                           const DecoderLayerConfig& config,
                           LayerCache& cache,
                           Tensor& out,
                           aclrtStream stream);

// Single-token decode. hidden is [1, H]. pos is the absolute position of
// this token (drives RoPE). cache_len is the number of valid K/V rows
// already in the cache; this call writes row `cache_len`. Caller bumps
// seq_len after all 32 layers run.
void decoder_layer_step(const Tensor& hidden,
                        const DecoderLayerWeights& weights,
                        const Tensor& cos_table,
                        const Tensor& sin_table,
                        int32_t pos,
                        int64_t cache_len,
                        const DecoderLayerConfig& config,
                        LayerCache& cache,
                        Tensor& out,
                        aclrtStream stream);

LayerCache make_layer_cache(int64_t max_seq_len,
                            const DecoderLayerConfig& config,
                            aclrtStream stream);

}  // namespace hy_mt2
