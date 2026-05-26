#include "kernel_operator.h"

using namespace AscendC;

class KernelRopeFullCustom {
public:
    __aicore__ inline KernelRopeFullCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR rowMap, GM_ADDR out,
                                uint32_t rows, uint32_t dim, uint32_t halfDim) {
        this->row = GetBlockIdx();
        this->rows = rows;
        this->dim = dim;
        this->halfDim = halfDim;

        xGm.SetGlobalBuffer((__gm__ half*)x, static_cast<uint64_t>(rows) * dim);
        cosGm.SetGlobalBuffer((__gm__ half*)cos);
        sinGm.SetGlobalBuffer((__gm__ half*)sin);
        rowMapGm.SetGlobalBuffer((__gm__ int32_t*)rowMap, rows);
        outGm.SetGlobalBuffer((__gm__ half*)out, static_cast<uint64_t>(rows) * dim);

        pipe.InitBuffer(x1Fp16Buf, MAX_HALF_DIM * sizeof(half));
        pipe.InitBuffer(x2Fp16Buf, MAX_HALF_DIM * sizeof(half));
        pipe.InitBuffer(cosFp16Buf, MAX_HALF_DIM * sizeof(half));
        pipe.InitBuffer(sinFp16Buf, MAX_HALF_DIM * sizeof(half));
        pipe.InitBuffer(x1Fp32Buf, MAX_HALF_DIM * sizeof(float));
        pipe.InitBuffer(x2Fp32Buf, MAX_HALF_DIM * sizeof(float));
        pipe.InitBuffer(cosFp32Buf, MAX_HALF_DIM * sizeof(float));
        pipe.InitBuffer(sinFp32Buf, MAX_HALF_DIM * sizeof(float));
        pipe.InitBuffer(tmp1Buf, MAX_HALF_DIM * sizeof(float));
        pipe.InitBuffer(tmp2Buf, MAX_HALF_DIM * sizeof(float));
        pipe.InitBuffer(yFp32Buf, MAX_HALF_DIM * sizeof(float));
        pipe.InitBuffer(yFp16Buf, MAX_HALF_DIM * sizeof(half));
    }

    __aicore__ inline void Process() {
        if (row >= rows || halfDim > MAX_HALF_DIM) return;
        const uint32_t alignedHalf = ((halfDim + ALIGN_FP16 - 1) / ALIGN_FP16) * ALIGN_FP16;
        const int32_t pos = rowMapGm.GetValue(row);

        LocalTensor<half> x1Fp16 = x1Fp16Buf.Get<half>();
        LocalTensor<half> x2Fp16 = x2Fp16Buf.Get<half>();
        LocalTensor<half> cosFp16 = cosFp16Buf.Get<half>();
        LocalTensor<half> sinFp16 = sinFp16Buf.Get<half>();
        LocalTensor<float> x1Fp32 = x1Fp32Buf.Get<float>();
        LocalTensor<float> x2Fp32 = x2Fp32Buf.Get<float>();
        LocalTensor<float> cosFp32 = cosFp32Buf.Get<float>();
        LocalTensor<float> sinFp32 = sinFp32Buf.Get<float>();
        LocalTensor<float> tmp1 = tmp1Buf.Get<float>();
        LocalTensor<float> tmp2 = tmp2Buf.Get<float>();
        LocalTensor<float> yFp32 = yFp32Buf.Get<float>();
        LocalTensor<half> yFp16 = yFp16Buf.Get<half>();

        const uint64_t xOff = static_cast<uint64_t>(row) * dim;
        const uint64_t tableOff = static_cast<uint64_t>(pos) * halfDim;
        DataCopy(x1Fp16, xGm[xOff], alignedHalf);
        DataCopy(x2Fp16, xGm[xOff + halfDim], alignedHalf);
        DataCopy(cosFp16, cosGm[tableOff], alignedHalf);
        DataCopy(sinFp16, sinGm[tableOff], alignedHalf);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        Cast(x1Fp32, x1Fp16, RoundMode::CAST_NONE, alignedHalf);
        Cast(x2Fp32, x2Fp16, RoundMode::CAST_NONE, alignedHalf);
        Cast(cosFp32, cosFp16, RoundMode::CAST_NONE, alignedHalf);
        Cast(sinFp32, sinFp16, RoundMode::CAST_NONE, alignedHalf);
        PipeBarrier<PIPE_V>();

        Mul(tmp1, x1Fp32, cosFp32, alignedHalf);
        Mul(tmp2, x2Fp32, sinFp32, alignedHalf);
        PipeBarrier<PIPE_V>();
        Sub(yFp32, tmp1, tmp2, alignedHalf);
        PipeBarrier<PIPE_V>();
        Cast(yFp16, yFp32, RoundMode::CAST_RINT, alignedHalf);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(outGm[xOff], yFp16, alignedHalf);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);

        Mul(tmp1, x2Fp32, cosFp32, alignedHalf);
        Mul(tmp2, x1Fp32, sinFp32, alignedHalf);
        PipeBarrier<PIPE_V>();
        Add(yFp32, tmp1, tmp2, alignedHalf);
        PipeBarrier<PIPE_V>();
        Cast(yFp16, yFp32, RoundMode::CAST_RINT, alignedHalf);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(outGm[xOff + halfDim], yFp16, alignedHalf);
    }

private:
    static constexpr uint32_t MAX_HALF_DIM = 128;
    static constexpr uint32_t ALIGN_FP16 = 16;

    uint32_t row;
    uint32_t rows;
    uint32_t dim;
    uint32_t halfDim;
    GlobalTensor<half> xGm;
    GlobalTensor<half> cosGm;
    GlobalTensor<half> sinGm;
    GlobalTensor<int32_t> rowMapGm;
    GlobalTensor<half> outGm;
    TPipe pipe;
    TBuf<TPosition::VECCALC> x1Fp16Buf;
    TBuf<TPosition::VECCALC> x2Fp16Buf;
    TBuf<TPosition::VECCALC> cosFp16Buf;
    TBuf<TPosition::VECCALC> sinFp16Buf;
    TBuf<TPosition::VECCALC> x1Fp32Buf;
    TBuf<TPosition::VECCALC> x2Fp32Buf;
    TBuf<TPosition::VECCALC> cosFp32Buf;
    TBuf<TPosition::VECCALC> sinFp32Buf;
    TBuf<TPosition::VECCALC> tmp1Buf;
    TBuf<TPosition::VECCALC> tmp2Buf;
    TBuf<TPosition::VECCALC> yFp32Buf;
    TBuf<TPosition::VECCALC> yFp16Buf;
};

extern "C" __global__ __aicore__ void rope_full_custom(GM_ADDR x,
                                                         GM_ADDR cos,
                                                         GM_ADDR sin,
                                                         GM_ADDR row_map,
                                                         GM_ADDR out,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelRopeFullCustom op;
    op.Init(x, cos, sin, row_map, out,
            tiling_data.rows, tiling_data.dim, tiling_data.halfDim);
    op.Process();
}
