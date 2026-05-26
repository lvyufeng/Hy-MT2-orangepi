# Hy-MT2 on Ascend 310B（Orange Pi AIPro 8T）

[English](README.md) · 中文

这是一个面向 [Tencent Hy-MT2-1.8B][hymt2] 的纯 C++/CANN 推理引擎，目标设备是 Orange Pi AIPro 8T / Ascend 310B4。

当前提供：

- `libhy_mt2.so`
- `hy_mt2_translate` CLI
- FP16 safetensors 加载，启动时 BF16→FP16
- 通过 [mlc-ai/tokenizers-cpp][tokcpp] 加载 HF `tokenizer.json`
- decode M=1 matmul 的 AscendC custom-op 快路径

本项目复用了 [`lvyufeng/minicpm-v-4.6-orangepi`][minicpmv] 中验证过的 C++/CANN 基础设施和自定义算子思路。

## 依赖

- Orange Pi AIPro 8T / Atlas 200I A2 / Ascend 310B4
- Ubuntu aarch64
- CANN 8.5.0：`/usr/local/Ascend/cann-8.5.0/aarch64-linux/`
- CMake >= 3.20
- Rust toolchain（用于 `tokenizers-cpp`）

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## 构建

```bash
source scripts/set_env.sh
./scripts/install_custom_ops.sh
./scripts/build.sh
```

## 下载模型

```bash
# 如果 Hugging Face 慢，可以使用 hf-mirror
HF_ENDPOINT=https://hf-mirror.com ./scripts/fetch_model.sh
```

模型会下载到 `./Hy-MT2-1.8B/`，权重目录不会进入 git。

## 运行

```bash
source scripts/set_env.sh

echo '我叫沃尔夫冈，我住在柏林。' | \
  ./build/hy_mt2_translate --model ./Hy-MT2-1.8B --tgt English --max-new 256 --stream
```

stdin 每行一条源文本，stdout 每行输出一条翻译。`--src` 目前只是为了兼容 CLI 形状保留，实际 prompt 只使用 `--tgt`。

## 性能

Orange Pi AIPro 8T 上当前 FP16 baseline：

| bench | setting | result |
|---|---|---:|
| prefill | prompt_len=8 | 22.74 s · 0.35 tok/s |
| decode | prompt_len=8, decode=30 | 686.39 ms/token · 1.46 tok/s |
| lm_head | 15 × 8192 cube chunks | 37.67 ms/pass |

custom cube 路径让常见 decode M=1 projection/MLP matmul 相比原 public-aclnn transposed-view 路径快约 40-50x，custom decode-attention 路径把当前 30-token decode bench 提到约 1.46 tok/s。当前主要瓶颈还在 RoPE/RMSNorm 的 launch 开销、临时 tensor 拷贝，以及 FP16 权重带宽。

常用 benchmark：

```bash
source scripts/set_env.sh
./build/bench_prefill --model ./Hy-MT2-1.8B --prompt-len 8 --iters 1
./build/bench_decode --model ./Hy-MT2-1.8B --prompt-len 8 --decode 30
./build/bench_lm_head --iters 5
./build/bench_matmul_throughput --iters 20
```

## 测试

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

## 当前状态

已跑通：

- 模型权重加载
- tokenizer 加载
- decoder prefill/decode loop
- tied-embedding lm_head
- CLI 翻译路径
- 真实模型 benchmark 路径

下一步性能工作：

- 拆解每 token decode 耗时
- fuse/替换 RoPE 和 q/k RMSNorm
- 使用 custom 或隐藏 CANN incremental attention
- W8A16 / 更低 bit 权重量化

## License

本仓库代码与上游模型权重遵循 [Tencent HY Community License](./LICENSE)。

[hymt2]: https://huggingface.co/tencent/Hy-MT2-1.8B
[tokcpp]: https://github.com/mlc-ai/tokenizers-cpp
[minicpmv]: https://github.com/lvyufeng/minicpm-v-4.6-orangepi
