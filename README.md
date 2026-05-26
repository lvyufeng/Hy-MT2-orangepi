# Hy-MT2 on Ascend 310B (Orange Pi AIPro 8T)

[English] · [中文](README_ZH.md)

Pure C++/CANN inference engine for [Tencent Hy-MT2-1.8B][hymt2] on Orange Pi AIPro 8T / Ascend 310B4.

The runtime exposes:

- `libhy_mt2.so`
- `hy_mt2_translate` CLI
- FP16 safetensors loading with BF16→FP16 conversion
- HF `tokenizer.json` via [mlc-ai/tokenizers-cpp][tokcpp]
- AscendC custom-op path for decode M=1 matmuls

This project reuses infrastructure and custom-kernel patterns from [`lvyufeng/minicpm-v-4.6-orangepi`][minicpmv].

## Requirements

- Orange Pi AIPro 8T / Atlas 200I A2 / Ascend 310B4
- Ubuntu aarch64
- CANN 8.5.0 at `/usr/local/Ascend/cann-8.5.0/aarch64-linux/`
- CMake >= 3.20
- Rust toolchain for `tokenizers-cpp`

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Build

```bash
source scripts/set_env.sh
./scripts/install_custom_ops.sh
./scripts/build.sh
```

## Download model

```bash
# If Hugging Face is slow from your network, use HF_ENDPOINT=https://hf-mirror.com
HF_ENDPOINT=https://hf-mirror.com ./scripts/fetch_model.sh
```

The model is downloaded to `./Hy-MT2-1.8B/`. Weights are ignored by git.

## Run

```bash
source scripts/set_env.sh

echo '我叫沃尔夫冈，我住在柏林。' | \
  ./build/hy_mt2_translate --model ./Hy-MT2-1.8B --tgt English --max-new 256 --stream
```

Input is one source sentence per stdin line. `--src` is accepted for CLI compatibility; the current prompt only uses `--tgt`.

## Benchmarks

On Orange Pi AIPro 8T, current FP16 baseline:

| bench | setting | result |
|---|---|---:|
| prefill | prompt_len=8 | 22.74 s · 0.35 tok/s |
| decode | prompt_len=8, decode=30 | 1.28 s/token · 0.78 tok/s |
| lm_head | 15 × 8192 cube chunks | 37.67 ms/pass |

The custom cube path improves common decode M=1 projection/MLP matmuls by roughly 40-50x versus the original public-aclnn transposed-view path. The remaining bottlenecks are attention/RoPE/RMSNorm launch overheads, temporary tensor copies, and the still-FP16 weight bandwidth.

Useful benchmark commands:

```bash
source scripts/set_env.sh
./build/bench_prefill --model ./Hy-MT2-1.8B --prompt-len 8 --iters 1
./build/bench_decode --model ./Hy-MT2-1.8B --prompt-len 8 --decode 30
./build/bench_lm_head --iters 5
./build/bench_matmul_throughput --iters 20
```

## Tests

```bash
source scripts/set_env.sh
./build/test_smoke
./build/test_weights_load
./build/test_ops_baseline
./build/test_matmul_cube
./build/test_decoder_layer
./build/test_language_model
HY_MT2_TOKENIZER_JSON=./Hy-MT2-1.8B/tokenizer.json ./build/test_tokenizer
```

## Current status

Working:

- model weight loading
- tokenizer loading
- decoder prefill/decode loop
- tied-embedding lm_head
- CLI translation path
- real-model benchmark path

Next performance work:

- profile per-token decode breakdown
- fuse/replace RoPE and q/k RMSNorm
- use custom or hidden CANN incremental attention
- W8A16 / lower-bit quantized weights

## License

This repo and the upstream model weights follow the [Tencent HY Community License](./LICENSE).

[hymt2]: https://huggingface.co/tencent/Hy-MT2-1.8B
[tokcpp]: https://github.com/mlc-ai/tokenizers-cpp
[minicpmv]: https://github.com/lvyufeng/minicpm-v-4.6-orangepi
