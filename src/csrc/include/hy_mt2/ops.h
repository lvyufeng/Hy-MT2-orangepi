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
void argmax_last_dim(const Tensor& self, Tensor& out, aclrtStream stream);

void add(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);
void mul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);
void silu(const Tensor& self, Tensor& out, aclrtStream stream);
void softmax_last_dim(const Tensor& self, Tensor& out, aclrtStream stream);
void rms_norm(const Tensor& x, const Tensor& gamma, Tensor& out, double epsilon, aclrtStream stream);
void cast(const Tensor& self, Tensor& out, aclrtStream stream);

void batch_matmul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);
void permute(const Tensor& self, const std::vector<int64_t>& dims, Tensor& out, aclrtStream stream);
void muls(const Tensor& self, float scalar, Tensor& out, aclrtStream stream);
void mean(const Tensor& self, const std::vector<int64_t>& dims, bool keep_dim, Tensor& out, aclrtStream stream);

}  // namespace hy_mt2
