#include "hy_mt2/ops.h"
#include "hy_mt2/acl_context.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_batch_matmul.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_embedding.h>
#include <aclnnop/aclnn_mean.h>
#include <aclnnop/aclnn_mm.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_permute.h>
#include <aclnnop/aclnn_rsqrt.h>
#include <aclnnop/aclnn_sigmoid.h>
#include <aclnnop/aclnn_silu.h>
#include <aclnnop/aclnn_softmax.h>
#include <aclnnop/aclnn_sub.h>

#if __has_include(<aclnn_matmul_cube_custom.h>)
#include <aclnn_matmul_cube_custom.h>
#define HY_MT2_HAS_MATMUL_CUBE_CUSTOM 1
#else
#define HY_MT2_HAS_MATMUL_CUBE_CUSTOM 0
#endif

namespace hy_mt2 {
namespace {

struct AclTensorHandle {
    aclTensor* tensor{nullptr};
    std::vector<int64_t> view_dims;
    std::vector<int64_t> strides;
    std::vector<int64_t> storage_dims;

    ~AclTensorHandle() { if (tensor) aclDestroyTensor(tensor); }
    AclTensorHandle() = default;
    AclTensorHandle(const AclTensorHandle&) = delete;
    AclTensorHandle& operator=(const AclTensorHandle&) = delete;
};

void make_acl_tensor(const Tensor& t, AclTensorHandle& h) {
    h.view_dims = t.shape();
    h.storage_dims = t.shape();
    h.strides.assign(t.shape().size(), 1);
    for (int i = static_cast<int>(t.shape().size()) - 2; i >= 0; --i) {
        h.strides[i] = h.strides[i + 1] * t.shape()[i + 1];
    }
    h.tensor = aclCreateTensor(
        h.view_dims.data(), h.view_dims.size(), to_acl_dtype(t.dtype()),
        h.strides.data(), 0, ACL_FORMAT_ND,
        h.storage_dims.data(), h.storage_dims.size(), t.data());
    if (h.tensor == nullptr) throw std::runtime_error("aclCreateTensor returned null");
}

void run_op(const char* name,
            uint64_t ws_size,
            aclOpExecutor* executor,
            aclrtStream stream,
            aclnnStatus (*launch)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
    void* workspace = nullptr;
    if (ws_size > 0) {
        check_acl(aclrtMalloc(&workspace, ws_size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace");
        check_acl(aclrtMemsetAsync(workspace, ws_size, 0, ws_size, stream), "aclrtMemsetAsync workspace");
    }
    auto ret = launch(workspace, ws_size, executor, stream);
    if (ret != 0) {
        if (workspace) aclrtFree(workspace);
        throw std::runtime_error(std::string(name) + " failed: " + std::to_string(ret));
    }
    auto sync_ret = aclrtSynchronizeStream(stream);
    if (workspace) aclrtFree(workspace);
    check_acl(sync_ret, "aclrtSynchronizeStream");
}

void check_same_shape(const Tensor& a, const Tensor& b, const char* op) {
    if (a.shape() != b.shape()) throw std::runtime_error(std::string(op) + " shape mismatch");
}

}  // namespace

void embedding_lookup(const Tensor& weight,
                      const std::vector<int32_t>& host_ids,
                      Tensor& out,
                      aclrtStream stream) {
    if (weight.shape().size() != 2 || out.shape().size() != 2) {
        throw std::runtime_error("embedding weight/out must be 2D");
    }
    if (out.shape()[0] != static_cast<int64_t>(host_ids.size()) || out.shape()[1] != weight.shape()[1]) {
        throw std::runtime_error("embedding out shape mismatch");
    }

    Tensor ids({static_cast<int64_t>(host_ids.size())}, DType::Int32);
    ids.allocate();
    ids.copy_from_host(host_ids.data(), host_ids.size() * sizeof(int32_t));

    AclTensorHandle hw, hi, ho;
    make_acl_tensor(weight, hw);
    make_acl_tensor(ids, hi);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnEmbeddingGetWorkspaceSize(hw.tensor, hi.tensor, ho.tensor, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnEmbeddingGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnEmbedding", ws_size, executor, stream, aclnnEmbedding);
}

void matmul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream) {
    if (a.shape().size() != 2 || b.shape().size() != 2 || out.shape().size() != 2) {
        throw std::runtime_error("matmul tensors must be 2D");
    }
    if (a.shape()[1] != b.shape()[0] || out.shape() != std::vector<int64_t>{a.shape()[0], b.shape()[1]}) {
        throw std::runtime_error("matmul shape mismatch");
    }
    AclTensorHandle ha, hb, ho;
    make_acl_tensor(a, ha);
    make_acl_tensor(b, hb);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    constexpr int8_t kCubeMathType = 1;
    auto ret = aclnnMmGetWorkspaceSize(ha.tensor, hb.tensor, ho.tensor, kCubeMathType, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnMmGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnMm", ws_size, executor, stream, aclnnMm);
}

void matmul_b_transposed(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream) {
    if (a.shape().size() != 2 || b.shape().size() != 2 || out.shape().size() != 2) {
        throw std::runtime_error("matmul_b_transposed tensors must be 2D");
    }
    const int64_t m = a.shape()[0];
    const int64_t k = a.shape()[1];
    if (b.shape()[1] != k || out.shape() != std::vector<int64_t>{m, b.shape()[0]}) {
        throw std::runtime_error("matmul_b_transposed shape mismatch");
    }

    AclTensorHandle ha, hb, ho;
    make_acl_tensor(a, ha);
    make_acl_tensor(out, ho);

    hb.storage_dims = b.shape();
    hb.view_dims = {b.shape()[1], b.shape()[0]};
    hb.strides = {1, b.shape()[1]};
    hb.tensor = aclCreateTensor(hb.view_dims.data(), hb.view_dims.size(), to_acl_dtype(b.dtype()),
                                hb.strides.data(), 0, ACL_FORMAT_ND,
                                hb.storage_dims.data(), hb.storage_dims.size(), b.data());
    if (hb.tensor == nullptr) throw std::runtime_error("aclCreateTensor returned null for transposed B");

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    constexpr int8_t kCubeMathType = 1;
    auto ret = aclnnMmGetWorkspaceSize(ha.tensor, hb.tensor, ho.tensor, kCubeMathType, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnMmGetWorkspaceSize B^T failed: " + std::to_string(ret));
    run_op("aclnnMm_Bt", ws_size, executor, stream, aclnnMm);
}

void matmul_b_natural(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream) {
    if (a.shape().size() != 2 || b.shape().size() != 2 || out.shape().size() != 2) {
        throw std::runtime_error("matmul_b_natural tensors must be 2D");
    }
    const int64_t m = a.shape()[0];
    const int64_t k = a.shape()[1];
    const int64_t n = b.shape()[1];
    if (b.shape()[0] != k || out.shape() != std::vector<int64_t>{m, n}) {
        throw std::runtime_error("matmul_b_natural shape mismatch");
    }
#if HY_MT2_HAS_MATMUL_CUBE_CUSTOM
    if (m == 1 && a.dtype() == DType::Float16 && b.dtype() == DType::Float16 && out.dtype() == DType::Float16 &&
        n <= 16384 && (n % 128) == 0) {
        AclTensorHandle ha, hb, ho;
        make_acl_tensor(a, ha);
        make_acl_tensor(b, hb);
        make_acl_tensor(out, ho);
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnMatmulCubeCustomGetWorkspaceSize(ha.tensor, hb.tensor, ho.tensor, &ws_size, &executor);
        if (ret != 0) {
            throw std::runtime_error("aclnnMatmulCubeCustomGetWorkspaceSize failed: " + std::to_string(ret));
        }
        run_op("aclnnMatmulCubeCustom", ws_size, executor, stream, aclnnMatmulCubeCustom);
        return;
    }
#endif
    matmul(a, b, out, stream);
}

void argmax_last_dim(const Tensor& self, Tensor& out, aclrtStream stream) {
    if (self.shape().empty()) throw std::runtime_error("argmax input must have rank >= 1");
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnArgMaxGetWorkspaceSize(hs.tensor, static_cast<int64_t>(self.shape().size() - 1), false,
                                           ho.tensor, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnArgMaxGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnArgMax", ws_size, executor, stream, aclnnArgMax);
}

void add(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream) {
    check_same_shape(a, b, "add");
    check_same_shape(a, out, "add");
    AclTensorHandle ha, hb, ho;
    make_acl_tensor(a, ha);
    make_acl_tensor(b, hb);
    make_acl_tensor(out, ho);
    float alpha_value = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alpha_value, ACL_FLOAT);
    if (!alpha) throw std::runtime_error("aclCreateScalar(alpha) failed");
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(ha.tensor, hb.tensor, alpha, ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        aclDestroyScalar(alpha);
        throw std::runtime_error("aclnnAddGetWorkspaceSize failed: " + std::to_string(ret));
    }
    try { run_op("aclnnAdd", ws_size, executor, stream, aclnnAdd); }
    catch (...) { aclDestroyScalar(alpha); throw; }
    aclDestroyScalar(alpha);
}

void mul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream) {
    check_same_shape(a, out, "mul");
    AclTensorHandle ha, hb, ho;
    make_acl_tensor(a, ha);
    make_acl_tensor(b, hb);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(ha.tensor, hb.tensor, ho.tensor, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnMulGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnMul", ws_size, executor, stream, aclnnMul);
}

void silu(const Tensor& self, Tensor& out, aclrtStream stream) {
    check_same_shape(self, out, "silu");
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnSiluGetWorkspaceSize(hs.tensor, ho.tensor, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnSiluGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnSilu", ws_size, executor, stream, aclnnSilu);
}

void sigmoid(const Tensor& self, Tensor& out, aclrtStream stream) {
    check_same_shape(self, out, "sigmoid");
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnSigmoidGetWorkspaceSize(hs.tensor, ho.tensor, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnSigmoidGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnSigmoid", ws_size, executor, stream, aclnnSigmoid);
}

void sub(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream) {
    check_same_shape(a, out, "sub");
    AclTensorHandle ha, hb, ho;
    make_acl_tensor(a, ha);
    make_acl_tensor(b, hb);
    make_acl_tensor(out, ho);
    float alpha_value = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alpha_value, ACL_FLOAT);
    if (!alpha) throw std::runtime_error("aclCreateScalar(alpha) failed");
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnSubGetWorkspaceSize(ha.tensor, hb.tensor, alpha, ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        aclDestroyScalar(alpha);
        throw std::runtime_error("aclnnSubGetWorkspaceSize failed: " + std::to_string(ret));
    }
    try { run_op("aclnnSub", ws_size, executor, stream, aclnnSub); }
    catch (...) { aclDestroyScalar(alpha); throw; }
    aclDestroyScalar(alpha);
}

void softmax_last_dim(const Tensor& self, Tensor& out, aclrtStream stream) {
    check_same_shape(self, out, "softmax");
    if (self.shape().empty()) throw std::runtime_error("softmax input must have rank >= 1");
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnSoftmaxGetWorkspaceSize(hs.tensor, static_cast<int64_t>(self.shape().size() - 1),
                                            ho.tensor, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnSoftmaxGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnSoftmax", ws_size, executor, stream, aclnnSoftmax);
}

void rms_norm(const Tensor& x, const Tensor& gamma, Tensor& out, double epsilon, aclrtStream stream) {
    if (x.shape() != out.shape() || x.shape().empty()) throw std::runtime_error("rms_norm x/out shape mismatch");
    if (gamma.shape() != std::vector<int64_t>{x.shape().back()}) throw std::runtime_error("rms_norm gamma shape mismatch");
    if (x.dtype() != DType::Float16 || gamma.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("rms_norm requires fp16 x/gamma/out");
    }

    auto reduce_shape = x.shape();
    reduce_shape.back() = 1;

    Tensor x_sq(x.shape(), DType::Float16); x_sq.allocate();
    Tensor mean_x_sq(reduce_shape, DType::Float16); mean_x_sq.allocate();
    Tensor rstd(reduce_shape, DType::Float16); rstd.allocate();
    Tensor scaled(x.shape(), DType::Float16); scaled.allocate();

    mul(x, x, x_sq, stream);
    mean(x_sq, {static_cast<int64_t>(x.shape().size() - 1)}, true, mean_x_sq, stream);

    {
        AclTensorHandle hmean, hrstd;
        make_acl_tensor(mean_x_sq, hmean);
        make_acl_tensor(rstd, hrstd);
        float eps_f = static_cast<float>(epsilon);
        aclScalar* eps_scalar = aclCreateScalar(&eps_f, ACL_FLOAT);
        float alpha_f = 1.0f;
        aclScalar* alpha_scalar = aclCreateScalar(&alpha_f, ACL_FLOAT);
        if (!eps_scalar || !alpha_scalar) throw std::runtime_error("rms_norm scalar allocation failed");
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnAddsGetWorkspaceSize(hmean.tensor, eps_scalar, alpha_scalar,
                                             hrstd.tensor, &ws_size, &executor);
        if (ret != 0) {
            aclDestroyScalar(eps_scalar);
            aclDestroyScalar(alpha_scalar);
            throw std::runtime_error("rms_norm Adds ws failed: " + std::to_string(ret));
        }
        try { run_op("rms_norm Adds", ws_size, executor, stream, aclnnAdds); }
        catch (...) { aclDestroyScalar(eps_scalar); aclDestroyScalar(alpha_scalar); throw; }
        aclDestroyScalar(eps_scalar);
        aclDestroyScalar(alpha_scalar);
    }

    {
        AclTensorHandle hin, hout;
        make_acl_tensor(rstd, hin);
        make_acl_tensor(rstd, hout);
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnRsqrtGetWorkspaceSize(hin.tensor, hout.tensor, &ws_size, &executor);
        if (ret != 0) throw std::runtime_error("rms_norm Rsqrt ws failed: " + std::to_string(ret));
        run_op("rms_norm Rsqrt", ws_size, executor, stream, aclnnRsqrt);
    }

    mul(x, rstd, scaled, stream);
    mul(scaled, gamma, out, stream);
}

void cast(const Tensor& self, Tensor& out, aclrtStream stream) {
    check_same_shape(self, out, "cast");
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnCastGetWorkspaceSize(hs.tensor, to_acl_dtype(out.dtype()), ho.tensor, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnCastGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnCast", ws_size, executor, stream, aclnnCast);
}

void batch_matmul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream) {
    AclTensorHandle ha, hb, ho;
    make_acl_tensor(a, ha);
    make_acl_tensor(b, hb);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    constexpr int8_t kCubeMathType = 1;
    auto ret = aclnnBatchMatMulGetWorkspaceSize(ha.tensor, hb.tensor, ho.tensor, kCubeMathType, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnBatchMatMulGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnBatchMatMul", ws_size, executor, stream, aclnnBatchMatMul);
}

void permute(const Tensor& self, const std::vector<int64_t>& dims, Tensor& out, aclrtStream stream) {
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    aclIntArray* dims_arr = aclCreateIntArray(dims.data(), dims.size());
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPermuteGetWorkspaceSize(hs.tensor, dims_arr, ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        aclDestroyIntArray(dims_arr);
        throw std::runtime_error("aclnnPermuteGetWorkspaceSize failed: " + std::to_string(ret));
    }
    try { run_op("aclnnPermute", ws_size, executor, stream, aclnnPermute); }
    catch (...) { aclDestroyIntArray(dims_arr); throw; }
    aclDestroyIntArray(dims_arr);
}

void muls(const Tensor& self, float scalar, Tensor& out, aclrtStream stream) {
    check_same_shape(self, out, "muls");
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    aclScalar* s = aclCreateScalar(&scalar, ACL_FLOAT);
    if (!s) throw std::runtime_error("muls aclCreateScalar failed");
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulsGetWorkspaceSize(hs.tensor, s, ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        aclDestroyScalar(s);
        throw std::runtime_error("aclnnMulsGetWorkspaceSize failed: " + std::to_string(ret));
    }
    try { run_op("aclnnMuls", ws_size, executor, stream, aclnnMuls); }
    catch (...) { aclDestroyScalar(s); throw; }
    aclDestroyScalar(s);
}

void mean(const Tensor& self, const std::vector<int64_t>& dims, bool keep_dim, Tensor& out, aclrtStream stream) {
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    aclIntArray* dim_arr = aclCreateIntArray(dims.data(), dims.size());
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMeanGetWorkspaceSize(hs.tensor, dim_arr, keep_dim, to_acl_dtype(self.dtype()), ho.tensor,
                                         &ws_size, &executor);
    if (ret != 0) {
        aclDestroyIntArray(dim_arr);
        throw std::runtime_error("aclnnMeanGetWorkspaceSize failed: " + std::to_string(ret));
    }
    try { run_op("aclnnMean", ws_size, executor, stream, aclnnMean); }
    catch (...) { aclDestroyIntArray(dim_arr); throw; }
    aclDestroyIntArray(dim_arr);
}

void apply_rope_full(const Tensor& x,
                     const Tensor& cos_table,
                     const Tensor& sin_table,
                     const std::vector<int32_t>& row_to_t,
                     Tensor& out,
                     aclrtStream stream) {
    if (x.dtype() != DType::Float16 || cos_table.dtype() != DType::Float16 ||
        sin_table.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("apply_rope_full requires fp16 tensors");
    }
    if (x.shape().size() != 2 || out.shape() != x.shape()) {
        throw std::runtime_error("apply_rope_full expects x/out shape [N, D]");
    }
    if (cos_table.shape().size() != 2 || sin_table.shape() != cos_table.shape()) {
        throw std::runtime_error("apply_rope_full cos/sin must have shape [T, D/2]");
    }

    const int64_t N = x.shape()[0];
    const int64_t D = x.shape()[1];
    if (D <= 0 || D % 2 != 0) throw std::runtime_error("apply_rope_full D must be positive and even");
    const int64_t Half = D / 2;
    if (cos_table.shape()[1] != Half) throw std::runtime_error("apply_rope_full table width mismatch");
    if (static_cast<int64_t>(row_to_t.size()) != N) throw std::runtime_error("apply_rope_full row_to_t size mismatch");
    for (int32_t t : row_to_t) {
        if (t < 0 || t >= cos_table.shape()[0]) throw std::runtime_error("apply_rope_full row_to_t out of range");
    }

    Tensor x1({N, Half}, DType::Float16); x1.allocate();
    Tensor x2({N, Half}, DType::Float16); x2.allocate();
    Tensor cos_e({N, Half}, DType::Float16); cos_e.allocate();
    Tensor sin_e({N, Half}, DType::Float16); sin_e.allocate();
    Tensor a({N, Half}, DType::Float16); a.allocate();
    Tensor b({N, Half}, DType::Float16); b.allocate();
    Tensor y1({N, Half}, DType::Float16); y1.allocate();
    Tensor y2({N, Half}, DType::Float16); y2.allocate();

    const size_t elem = dtype_size(DType::Float16);
    const size_t row_bytes = static_cast<size_t>(D) * elem;
    const size_t half_bytes = static_cast<size_t>(Half) * elem;
    auto* x_base = static_cast<const uint8_t*>(x.data());
    auto* cos_base = static_cast<const uint8_t*>(cos_table.data());
    auto* sin_base = static_cast<const uint8_t*>(sin_table.data());
    auto* out_base = static_cast<uint8_t*>(out.data());

    for (int64_t n = 0; n < N; ++n) {
        const uint8_t* x_row = x_base + static_cast<size_t>(n) * row_bytes;
        check_acl(aclrtMemcpyAsync(static_cast<uint8_t*>(x1.data()) + static_cast<size_t>(n) * half_bytes,
                                   half_bytes, x_row, half_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "apply_rope_full copy x1");
        check_acl(aclrtMemcpyAsync(static_cast<uint8_t*>(x2.data()) + static_cast<size_t>(n) * half_bytes,
                                   half_bytes, x_row + half_bytes, half_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "apply_rope_full copy x2");
        const int64_t t = row_to_t[n];
        check_acl(aclrtMemcpyAsync(static_cast<uint8_t*>(cos_e.data()) + static_cast<size_t>(n) * half_bytes,
                                   half_bytes,
                                   cos_base + static_cast<size_t>(t) * half_bytes,
                                   half_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "apply_rope_full copy cos");
        check_acl(aclrtMemcpyAsync(static_cast<uint8_t*>(sin_e.data()) + static_cast<size_t>(n) * half_bytes,
                                   half_bytes,
                                   sin_base + static_cast<size_t>(t) * half_bytes,
                                   half_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "apply_rope_full copy sin");
    }
    check_acl(aclrtSynchronizeStream(stream), "apply_rope_full gather sync");

    mul(x1, cos_e, a, stream);
    mul(x2, sin_e, b, stream);
    sub(a, b, y1, stream);
    mul(x2, cos_e, a, stream);
    mul(x1, sin_e, b, stream);
    add(a, b, y2, stream);

    for (int64_t n = 0; n < N; ++n) {
        uint8_t* o_row = out_base + static_cast<size_t>(n) * row_bytes;
        check_acl(aclrtMemcpyAsync(o_row, half_bytes,
                                   static_cast<uint8_t*>(y1.data()) + static_cast<size_t>(n) * half_bytes,
                                   half_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "apply_rope_full scatter y1");
        check_acl(aclrtMemcpyAsync(o_row + half_bytes, half_bytes,
                                   static_cast<uint8_t*>(y2.data()) + static_cast<size_t>(n) * half_bytes,
                                   half_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "apply_rope_full scatter y2");
    }
    check_acl(aclrtSynchronizeStream(stream), "apply_rope_full scatter sync");
}

}  // namespace hy_mt2
