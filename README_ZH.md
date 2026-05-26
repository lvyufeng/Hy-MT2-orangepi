# Hy-MT2 on Ascend 310B（Orange Pi AIPro 20T）

[English](README.md) · 中文

这是一个面向 [Tencent Hy-MT2-1.8B][hymt2] 的纯 C++/CANN 推理引擎，目标是在
Orange Pi AIPro 20T 的 Ascend 310B4 NPU 上跑通并压榨性能。**计算路径全部在 NPU +
C++ 引擎中完成，tokenizer 使用 [mlc-ai/tokenizers-cpp][tokcpp] 包装 HF tokenizer.json，
热路径不依赖 Python。**

> 状态（Phase 1）：脚手架、NPU runtime smoke test、safetensors 元数据加载器（含 BF16→FP16
> device load）和 `tokenizers-cpp` wrapper 已完成。自定义算子、decoder layer、完整 LM 会在 Phase 2+ 落地。

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

## Quickstart（Phase 1）

```bash
source scripts/set_env.sh
./scripts/build.sh

./build/test_smoke
./build/test_weights_load

# 需要 tokenizer.json；可先跑 scripts/fetch_model.sh，或直接指向已有文件：
HY_MT2_TOKENIZER_JSON=/path/to/tokenizer.json ./build/test_tokenizer
```

期望核心输出：

```text
[ok] AclContext on device 0
[ok] Tensor H2D/D2H round-trip 4 fp16 values
[ok] WeightsIndex parses metadata and loads BF16->FP16 tensors
[ok] tokenizer loads Hy-MT2 tokenizer.json and encodes chat prompt
```

## Roadmap

| Phase | 内容 | 状态 |
|---|---|---|
| 0 | 仓库脚手架、AclContext/Tensor、NPU smoke test | done |
| 1 | safetensors loader（BF16→FP16）+ tokenizers-cpp | done |
| 2 | ops + AscendC 自定义算子（cube matmul、RMSNorm、SwiGLU、attention step、full RoPE、qk-norm） | |
| 3 | Hy-MT2 decoder layer + 完整 LM（32 层、GQA 16/4、tie embeddings） | |
| 4 | `Translator` API + `hy_mt2_translate` CLI | |
| 5 | benchmark + v1 性能优化 | |
| 6 | W8A16、官方 IFA/RoPE aclnn shim、GGUF Q4_0/1.25-bit | |

## License

本仓库代码与上游模型权重遵循 [Tencent HY Community License](./LICENSE)。请注意 LICENSE 中的
地域限制（不适用于欧盟区域）。

[hymt2]: https://huggingface.co/tencent/Hy-MT2-1.8B
[tokcpp]: https://github.com/mlc-ai/tokenizers-cpp
[minicpmv]: https://github.com/lvyufeng/minicpm-v-4.6-orangepi
