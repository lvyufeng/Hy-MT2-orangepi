#pragma once

#include "hy_mt2/tensor.h"

#include <acl/acl.h>

#include <cstdint>
#include <vector>

namespace hy_mt2 {

void embedding_lookup(const Tensor& weight,
                      const std::vector<int32_t>& host_ids,
                      Tensor& out,
                      aclrtStream stream);

void matmul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);
void matmul_b_transposed(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);
void matmul_b_natural(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);
bool has_attention_step_custom();
void attention_step_custom(const Tensor& query,
                           const Tensor& k_cache,
                           const Tensor& v_cache,
                           int64_t context,
                           int64_t num_q_heads,
                           int64_t num_kv_heads,
                           float scale,
                           Tensor& out,
                           aclrtStream stream);
bool has_rms_norm128_custom();
void rms_norm128_custom(const Tensor& x,
                        const Tensor& gamma,
                        double epsilon,
                        Tensor& out,
                        aclrtStream stream);
void argmax_last_dim(const Tensor& self, Tensor& out, aclrtStream stream);

void add(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);
void sub(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);
void mul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);
void silu(const Tensor& self, Tensor& out, aclrtStream stream);
void sigmoid(const Tensor& self, Tensor& out, aclrtStream stream);
void softmax_last_dim(const Tensor& self, Tensor& out, aclrtStream stream);
void rms_norm(const Tensor& x, const Tensor& gamma, Tensor& out, double epsilon, aclrtStream stream);
void cast(const Tensor& self, Tensor& out, aclrtStream stream);

void batch_matmul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);
void permute(const Tensor& self, const std::vector<int64_t>& dims, Tensor& out, aclrtStream stream);
void muls(const Tensor& self, float scalar, Tensor& out, aclrtStream stream);
void mean(const Tensor& self, const std::vector<int64_t>& dims, bool keep_dim, Tensor& out, aclrtStream stream);

// Full rotary positional embedding. x [N, head_dim] fp16; cos/sin tables
// [T, head_dim/2] fp16; row_to_t[n] maps row n to its absolute position.
// The rotation pairs elements (i, i+head_dim/2) — i.e. "half rotate" not
// "interleaved pairs". Matches Llama / Hy-MT2 RoPE convention.
void apply_rope_full(const Tensor& x,
                     const Tensor& cos_table,
                     const Tensor& sin_table,
                     const std::vector<int32_t>& row_to_t,
                     Tensor& out,
                     aclrtStream stream);

}  // namespace hy_mt2
