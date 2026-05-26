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

// Per-token scratch buffers reused across all decoder layers in a single
// decode step. Shapes are fixed by the model config and T=1, so a single
// instance can be reused for every layer in every decode token. Built via
// make_decoder_step_scratch().
struct DecoderStepScratch {
    Tensor normed;       // [1, hidden]
    Tensor q_full;       // [1, num_q_heads * head_dim]
    Tensor k_full;       // [1, num_kv_heads * head_dim]
    Tensor v_full;       // [1, num_kv_heads * head_dim]
    Tensor q_heads;      // [num_q_heads, head_dim]
    Tensor k_heads;      // [num_kv_heads, head_dim]
    Tensor q_normed;     // [num_q_heads, head_dim]
    Tensor k_normed;     // [num_kv_heads, head_dim]
    Tensor q_rope;       // [num_q_heads, head_dim]
    Tensor k_rope;       // [num_kv_heads, head_dim]
    Tensor attn_out;     // [1, num_q_heads * head_dim]
    Tensor attn_proj;    // [1, hidden]
    Tensor after_attn;   // [1, hidden]
    Tensor mlp_in;       // [1, hidden]
    Tensor gate;         // [1, intermediate]
    Tensor up;           // [1, intermediate]
    Tensor gate_act;     // [1, intermediate]
    Tensor gated;        // [1, intermediate]
    Tensor mlp_out;      // [1, hidden]
    Tensor q_row_map;    // [num_q_heads] int32
    Tensor k_row_map;    // [num_kv_heads] int32
};

DecoderStepScratch make_decoder_step_scratch(int64_t hidden_size,
                                             int64_t intermediate_size,
                                             const DecoderLayerConfig& config);

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
//
// If `scratch` is non-null, all per-layer intermediate tensors are read
// from it instead of being freshly allocated; the caller is expected to
// keep the scratch alive across all layers in a decode step.
void decoder_layer_step(const Tensor& hidden,
                        const DecoderLayerWeights& weights,
                        const Tensor& cos_table,
                        const Tensor& sin_table,
                        int32_t pos,
                        int64_t cache_len,
                        const DecoderLayerConfig& config,
                        LayerCache& cache,
                        Tensor& out,
                        aclrtStream stream,
                        DecoderStepScratch* scratch = nullptr);

LayerCache make_layer_cache(int64_t max_seq_len,
                            const DecoderLayerConfig& config,
                            aclrtStream stream);

}  // namespace hy_mt2
