# Hy-MT2 on Ascend 310B（Orange Pi AIPro 20T）

[English](README.md) · 中文

这是一个面向 [Tencent Hy-MT2-1.8B][hymt2] 的纯 C++/CANN 推理引擎，目标是在
Orange Pi AIPro 20T 的 Ascend 310B4 NPU 上跑通并压榨性能。**计算路径全部在 NPU +
C++ 引擎中完成，tokenizer 使用 [mlc-ai/tokenizers-cpp][tokcpp] 包装 HF tokenizer.json，
热路径不依赖 Python。**

> 状态（Phase 5）：v1 C++/CANN 路径已经打通到 `libhy_mt2.so`、`Translator`、
> `hy_mt2_translate`、NPU synthetic smoke tests、LM-head chunking，以及 prefill/decode/matmul
> benchmark binaries。下一步是 Phase 6 性能压榨（量化与 hidden aclnn shim）。

本仓库结构和我之前的 [`lvyufeng/minicpm-v-4.6-orangepi`][minicpmv] 保持一致，并会复用其中
验证过的 C++/AscendC 基础设施与自定义算子（cube matmul、RMSNorm、fused SwiGLU、attention step）。

## 依赖

- Orange Pi AIPro 20T / Atlas 200I A2 / Ascend 310B4
- Ubuntu 22.04 aarch64
- CANN 8.5.0：`/usr/local/Ascend/cann-8.5.0/aarch64-linux/`
  - 注意：本机没有常见的 `/usr/local/Ascend/ascend-toolkit/latest` symlink，构建时直接指向版本目录。
  - 可通过 `HY_MT2_ASCEND_TOOLKIT_ROOT` 或 `-DASCEND_TOOLKIT_ROOT=...` 覆盖。
- CMake ≥ 3.20
- Rust toolchain：后续接入 `tokenizers-cpp` 时需要，一次性安装即可。

## Quickstart（Phase 3）

```bash
source scripts/set_env.sh
./scripts/build.sh

./build/test_smoke
./build/test_weights_load
./build/test_ops_baseline
./build/test_decoder_layer
./build/test_language_model

# 需要 tokenizer.json；可先跑 scripts/fetch_model.sh，或直接指向已有文件：
HY_MT2_TOKENIZER_JSON=/path/to/tokenizer.json ./build/test_tokenizer
```

期望核心输出：

```text
[ok] AclContext on device 0
[ok] Tensor H2D/D2H round-trip 4 fp16 values
[ok] WeightsIndex parses metadata and loads BF16->FP16 tensors
[ok] tokenizer loads Hy-MT2 tokenizer.json and encodes chat prompt
[ok] baseline aclnn ops wrappers run on NPU
[ok] decoder layer prefill/step synthetic residual path
[ok] language model prefill/decode synthetic path
```

## CLI

```bash
source scripts/set_env.sh
./build/hy_mt2_translate --model ./Hy-MT2-1.8B --tgt English --max-new 256 --stream
```

stdin 每行一条源文本，stdout 每行输出一条翻译。`--src` 目前只是为了兼容 CLI 形状保留，实际 prompt 只需要目标语言。

## Benchmarks

```bash
source scripts/set_env.sh
./build/bench_matmul_throughput --iters 20
./build/bench_lm_head --iters 5
./build/bench_prefill --model ./Hy-MT2-1.8B --prompt-len 8 --iters 1
./build/bench_decode --model ./Hy-MT2-1.8B --prompt-len 8 --decode 30
```

真实 Hy-MT2-1.8B FP16 权重 baseline：

| bench | setting | result |
|---|---|---:|
| prefill | prompt_len=8 | 22.74 s · 0.35 tok/s |
| decode | prompt_len=8, decode=3 | 1.08 s/token · 0.92 tok/s |
| decode | prompt_len=8, decode=30 | 1.28 s/token · 0.78 tok/s |

当前 matmul microbench（本机 Orange Pi AIPro 20T，`--iters 1`，fp16 tensors）。`aclnn_bt` 是原 public-aclnn transposed-view 路径；`natural_or_cube` 使用预转置 `[K,N]` 权重，并在 M=1、N<=16384、N%128==0 时走 `MatmulCubeCustom`。

| path | M | N | K | ms/iter | GFLOP/s |
|---|---:|---:|---:|---:|---:|
| aclnn_bt | 1 | 2048 | 2048 | 27.14 | 0.31 |
| natural_or_cube | 1 | 2048 | 2048 | 0.64 | 13.08 |
| aclnn_bt | 1 | 512 | 2048 | 7.23 | 0.29 |
| natural_or_cube | 1 | 512 | 2048 | 0.16 | 12.84 |
| aclnn_bt | 1 | 6144 | 2048 | 79.96 | 0.31 |
| natural_or_cube | 1 | 6144 | 2048 | 1.69 | 14.88 |
| aclnn_bt | 1 | 2048 | 6144 | 80.94 | 0.31 |
| natural_or_cube | 1 | 2048 | 6144 | 1.60 | 15.69 |
| aclnn_bt | 1 | 120818 | 2048 | 362.20 | 1.37 |
| natural_or_cube | 1 | 120818 | 2048 | 351.42 | 1.41 |
| aclnn_bt | 16 | 2048 | 2048 | 0.71 | 188.61 |
| natural_or_cube | 16 | 2048 | 2048 | 0.67 | 200.09 |

decode projection/MLP 的 M=1 常见形状现在通过 cube path 提升约 40-50x。tied lm_head 现在切成 15 个 8192 列 cube chunk；`bench_lm_head --iters 3` 显示单次 greedy head pass **37.67 ms**，相比之前 full-vocab aclnn matmul microbench 的 ~351 ms 明显下降。

## Roadmap

| Phase | 内容 | 状态 |
|---|---|---|
| 0 | 仓库脚手架、AclContext/Tensor、NPU smoke test | done |
| 1 | safetensors loader（BF16→FP16）+ tokenizers-cpp | done |
| 2 | ops + AscendC 自定义算子 scaffold（cube matmul、RMSNorm、SwiGLU、attention step）+ baseline aclnn wrappers | done |
| 3 | Hy-MT2 decoder layer + 完整 LM（32 层、GQA 16/4、tie embeddings） | done |
| 4 | `Translator` API + `hy_mt2_translate` CLI | done |
| 5 | benchmark + v1 性能优化（LM-head chunking、prefill/decode/matmul benches） | done |
| 6 | W8A16、官方 IFA/RoPE aclnn shim、GGUF Q4_0/1.25-bit | |

## License

本仓库代码与上游模型权重遵循 [Tencent HY Community License](./LICENSE)。请注意 LICENSE 中的
地域限制（不适用于欧盟区域）。

[hymt2]: https://huggingface.co/tencent/Hy-MT2-1.8B
[tokcpp]: https://github.com/mlc-ai/tokenizers-cpp
[minicpmv]: https://github.com/lvyufeng/minicpm-v-4.6-orangepi
