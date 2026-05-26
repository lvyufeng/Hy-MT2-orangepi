# Hy-MT2 on Ascend 310B (Orange Pi AIPro 20T)

[English] · [中文](README_ZH.md)

A from-scratch C++/CANN inference engine for [Tencent Hy-MT2-1.8B][hymt2],
the 1.8B "fast-thinking" multilingual translation model from Hunyuan, on
the Ascend 310B4 NPU in the Orange Pi AIPro 20T board. **All compute runs
on the NPU through a single C++ engine — the tokenizer is wrapped via
[mlc-ai/tokenizers-cpp][tokcpp] (no Python on the hot path).**

> 🚧 **Status (Phase 5)**: v1 C++/CANN path is in place through
> `libhy_mt2.so`, `Translator`, `hy_mt2_translate`, synthetic NPU smoke tests,
> LM-head chunking, and prefill/decode/matmul benchmark binaries. The next
> work is Phase 6 performance squeeze (quantization and hidden aclnn shims).

This repo follows the same shape as my earlier
[`lvyufeng/minicpm-v-4.6-orangepi`][minicpmv]; several infrastructure
files and AscendC custom kernels (cube matmul, RMSNorm, fused SwiGLU,
attention step) are adapted from that work.

## Hardware & software prerequisites

- **Board**: Orange Pi AIPro 20T (or any Atlas 200I A2 / Ascend 310B4 device)
- **OS**: Ubuntu 22.04 aarch64
- **CANN toolkit**: 8.5.0 at `/usr/local/Ascend/cann-8.5.0/aarch64-linux/`
  (override with `HY_MT2_ASCEND_TOOLKIT_ROOT` or
  `-DASCEND_TOOLKIT_ROOT=...`). Note: the usual
  `ascend-toolkit/latest` symlink is **not** present on this stock image;
  point the build at the version directory directly.
- **CMake**: ≥ 3.20
- **Rust toolchain**: needed by `tokenizers-cpp` (the vendored HF
  tokenizer crate). One-time:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

## Quickstart (Phase 3 scope — runtime, ops, decoder, LM smoke tests)

```bash
source scripts/set_env.sh
./scripts/build.sh

./build/test_smoke
./build/test_weights_load
./build/test_ops_baseline
./build/test_decoder_layer
./build/test_language_model

# Requires tokenizer.json. Either fetch the model or point to a local file:
HY_MT2_TOKENIZER_JSON=/path/to/tokenizer.json ./build/test_tokenizer
```

Expected core outputs:

```
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

Input is one source sentence per stdin line. `--src` is accepted for CLI
compatibility, but the current Hy-MT2 prompt only needs the target language.

## Benchmarks

```bash
source scripts/set_env.sh
./build/bench_matmul_throughput --iters 20
./build/bench_lm_head --iters 5
./build/bench_prefill --model ./Hy-MT2-1.8B --prompt-len 8 --iters 1
./build/bench_decode --model ./Hy-MT2-1.8B --prompt-len 8 --decode 30
```

Real-model baseline on Hy-MT2-1.8B FP16 weights:

| bench | setting | result |
|---|---|---:|
| prefill | prompt_len=8 | 22.74 s · 0.35 tok/s |
| decode | prompt_len=8, decode=3 | 1.08 s/token · 0.92 tok/s |
| decode | prompt_len=8, decode=30 | 1.28 s/token · 0.78 tok/s |

Current matmul microbench snapshot on this Orange Pi AIPro 20T (`--iters 1`,
fp16 tensors). `aclnn_bt` is the original public-aclnn transposed-view path;
`natural_or_cube` uses pre-transposed `[K,N]` weights and routes M=1, N<=16384,
N%128==0 through `MatmulCubeCustom`.

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

The decode projection/MLP M=1 shapes are now roughly 40-50x faster with the
cube path. The tied lm_head now runs as 15 cube chunks of 8192 columns;
`bench_lm_head --iters 3` reports **37.67 ms** per greedy head pass, down from
~351 ms for the earlier full-vocab aclnn matmul microbench path.

## Repository layout

```
Hy-MT2-orangepi/
├── CMakeLists.txt
├── scripts/                       set_env / build / install_custom_ops / fetch_model
├── src/csrc/
│   ├── include/hy_mt2/            public headers
│   ├── lib/                       engine sources
│   ├── custom_ops/                AscendC custom kernels (Phase 2)
│   └── tools/                     CLI entry points (Phase 4)
├── tests/                         per-op / per-component tests
└── bench/                         microbenchmarks (Phase 5)
```

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo scaffold, AclContext + Tensor + smoke test | ✅ done |
| 1 | safetensors loader (BF16→FP16 cast) + tokenizers-cpp wrapper | ✅ done |
| 2 | Ops + custom AscendC kernels (cube matmul, RMSNorm, SwiGLU, attention-step scaffold; baseline public-aclnn wrappers) | ✅ done |
| 3 | Hy-MT2 decoder layer + full language model (32 layers, GQA 16/4, tie embeddings) | ✅ done |
| 4 | `Translator` API + `hy_mt2_translate` CLI | ✅ done |
| 5 | Bench + first perf rounds (lm_head chunking, prefill/decode/matmul benches) | ✅ done |
| 6 | W8A16 (aclnnQuantMatmulV3), IFA/RoPE aclnn shims, GGUF Q4_0 | |

## Hy-MT2 architecture (target)

- `HunYuanDenseV1ForCausalLM` — standard Llama-flavored dense decoder
- 32 layers · hidden 2048 · intermediate 6144 · SwiGLU
- GQA: 16 Q heads × 4 KV heads × head_dim 128
- Full RoPE (θ=10000, dynamic NTK scaling, max_position_embeddings 262144), `use_qk_norm=true`
- RMSNorm ε=1e-5, no attention/MLP bias
- `tie_word_embeddings=true` (lm_head shares embed_tokens)
- vocab 120818 · BOS 120000 · EOS 120020

## License

This repo (engine code) and the upstream model weights are licensed
under the [Tencent HY Community License](./LICENSE). See LICENSE for the
full terms, including the territory carve-out (excludes the EU).

[hymt2]: https://huggingface.co/tencent/Hy-MT2-1.8B
[tokcpp]: https://github.com/mlc-ai/tokenizers-cpp
[minicpmv]: https://github.com/lvyufeng/minicpm-v-4.6-orangepi
