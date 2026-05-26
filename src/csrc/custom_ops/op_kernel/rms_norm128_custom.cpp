#include "kernel_operator.h"

using namespace AscendC;

// Per-row RMSNorm specialized for hidden=128 fp16. One block per row; gamma
// loaded once per block. Pattern matches rms_norm1024_custom and reuses the
// FP32 accumulator path used by rope_full_custom.

class KernelRmsNorm128Custom {
public:
    __aicore__ inline KernelRmsNorm128Custom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR out,
                                uint32_t rows, float epsilon) {
        this->row = GetBlockIdx();
        this->rows = rows;
        this->epsilon = epsilon;

        xGm.SetGlobalBuffer((__gm__ half*)x, static_cast<uint64_t>(rows) * DIM);
        gammaGm.SetGlobalBuffer((__gm__ half*)gamma, DIM);
        outGm.SetGlobalBuffer((__gm__ half*)out, static_cast<uint64_t>(rows) * DIM);

        pipe.InitBuffer(xFp16Buf, DIM * sizeof(half));
        pipe.InitBuffer(gammaFp16Buf, DIM * sizeof(half));
        pipe.InitBuffer(xFp32Buf, DIM * sizeof(float));
        pipe.InitBuffer(gammaFp32Buf, DIM * sizeof(float));
        pipe.InitBuffer(sqBuf, DIM * sizeof(float));
        pipe.InitBuffer(outFp16Buf, DIM * sizeof(half));
        pipe.InitBuffer(reduceTmpBuf, DIM * sizeof(float));
        pipe.InitBuffer(reduceDstBuf, 32);
    }

    __aicore__ inline void Process() {
        if (row >= rows) return;

        LocalTensor<half> xFp16 = xFp16Buf.Get<half>();
        LocalTensor<half> gammaFp16 = gammaFp16Buf.Get<half>();
        LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
        LocalTensor<float> gammaFp32 = gammaFp32Buf.Get<float>();
        LocalTensor<float> sq = sqBuf.Get<float>();
        LocalTensor<half> outFp16 = outFp16Buf.Get<half>();
        LocalTensor<float> redTmp = reduceTmpBuf.Get<float>();
        LocalTensor<float> redDst = reduceDstBuf.Get<float>();

        DataCopy(gammaFp16, gammaGm[0], DIM);
        DataCopy(xFp16, xGm[static_cast<uint64_t>(row) * DIM], DIM);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        Cast(gammaFp32, gammaFp16, RoundMode::CAST_NONE, DIM);
        Cast(xFp32, xFp16, RoundMode::CAST_NONE, DIM);
        PipeBarrier<PIPE_V>();

        Mul(sq, xFp32, xFp32, DIM);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(redDst, sq, redTmp, DIM);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float sumSq = redDst.GetValue(0);
        const float invRms = 1.0f / sqrt(sumSq * INV_DIM + epsilon);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);

        Muls(sq, xFp32, invRms, DIM);
        PipeBarrier<PIPE_V>();
        Mul(xFp32, sq, gammaFp32, DIM);
        PipeBarrier<PIPE_V>();
        Cast(outFp16, xFp32, RoundMode::CAST_RINT, DIM);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(outGm[static_cast<uint64_t>(row) * DIM], outFp16, DIM);
    }

private:
    static constexpr uint32_t DIM = 128;
    static constexpr float INV_DIM = 1.0f / 128.0f;

    uint32_t row;
    uint32_t rows;
    float epsilon;
    GlobalTensor<half> xGm;
    GlobalTensor<half> gammaGm;
    GlobalTensor<half> outGm;
    TPipe pipe;
    TBuf<TPosition::VECCALC> xFp16Buf;
    TBuf<TPosition::VECCALC> gammaFp16Buf;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> gammaFp32Buf;
    TBuf<TPosition::VECCALC> sqBuf;
    TBuf<TPosition::VECCALC> outFp16Buf;
    TBuf<TPosition::VECCALC> reduceTmpBuf;
    TBuf<TPosition::VECCALC> reduceDstBuf;
};

extern "C" __global__ __aicore__ void rms_norm128_custom(GM_ADDR x, GM_ADDR gamma, GM_ADDR out,
                                                          GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelRmsNorm128Custom op;
    op.Init(x, gamma, out, tiling_data.rows, tiling_data.epsilon);
    op.Process();
}
