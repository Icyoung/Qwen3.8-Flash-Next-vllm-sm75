# 原生试跑环境与适配

本文记录已成功短测的条件，供开发者重建环境；不是零环境安装器，也不会安装或启动 systemd 服务。模型质量、32K 和长期散热尚未验收。完整过程见 [sm75-trial-20261014.md](sm75-trial-20261014.md)。

## 环境基线

- Ubuntu 24.04、4×Tesla T10 16GiB、8GB RAM、本地 SSD；实际使用总计 16GiB swap，服务未限制 MemoryMax/MemorySwapMax。早期 6GiB/no-swap 配置未通过。
- Python 3.12.15 独立 venv、Torch 2.13.0+cu129、vLLM 0.30.0+cu129、Transformers 5.15.1、FlashInfer 0.6.18。
- 引擎 overlay 固定 `d7c5d4204cc582bb9c05ff9b4e7a199ae3fbdb0c`，vLLM 基线固定 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`；先应用引擎文件，再应用本仓库补丁。
- 私有 CUDA 12.9（nvcc 12.9.86）、GCC13、curand 开发头 10.3.10.19；已编译 SM75 GDN 扩展与 FlashInfer 采样模块。仅有驱动不够完成这些编译。
- vLLM 的 FlashInfer 0.6.18.post1 依赖与固定 0.6.18 存在解析冲突，实测环境通过 overrides 处理；单用 constraints 不能解决。torchvision 使用匹配的 0.28.0+cu129；移除了 ABI 不匹配的可选 torchaudio/torchcodec。本次只验收短文本链路。

## 必要的运行适配

在上述固定引擎已安装的隔离 venv 中，从本仓库根目录执行：

```sh
VLLM_PACKAGE="$(python -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent)')"
python tools/install-ssd-overlay.py --package "$VLLM_PACKAGE"
python tools/install-lm-head-overlay.py --package "$VLLM_PACKAGE"
# 使用 MTP 时还需要修复草稿模型的量化输出头
python tools/install-lm-head-overlay.py --package "$VLLM_PACKAGE" --mtp-only

export PYTHONPATH="$PWD/runtime${PYTHONPATH:+:$PYTHONPATH}"
export Q38_INC_INT3=1
```

这些工具校验精确源 SHA-256；未知基线和已安装过的补丁会被拒绝。`runtime/sitecustomize.py` 在显式启用后为 API 及 spawn worker 安装 INT3/混合专家适配，失败会立即退出，避免悄悄回退。此桥接只针对本仓库的完整专家配方，不是通用 INT3 支持。

PLE 使用 `Q38_PLE_SSD=1`、`Q38_PLE_MANIFEST` 指向生成的 `ple-manifest.json`、`Q38_PLE_CACHE_BYTES=33554432`。开启实测 NCCL P2P 使用 `NCCL_P2P_DISABLE=0`、`NCCL_IB_DISABLE=1`；关闭 custom all-reduce 和 FlashInfer PCIe IPC。设置环境变量本身不代表路径已生效，实测通过 NCCL INFO 的 P2P/CUMEM 记录确认。

成功短测的核心参数为 TP4+EP4、FP16、FP8 KV、eager、max-model-len=2048、max-num-seqs=1、max-num-batched-tokens=64。每 rank KV 预算：无 MTP 128MiB、MTP3 256MiB、MTP6 512MiB。其余 QSA/GDN 参数需与固定引擎匹配；仓库中的 32K Docker profile 不能直接替代该测试条件。

## MTP 独立视图

目录应包含 `checkpoint-nomtp/`、`runtime-ssd/`、`ple/` 和 `mtp/model_extra_tensors.safetensors`。最后一个文件来自本次原始量化导出，SHA-256 固定为 `1aa640d89ba9764791a5d14cc5fa32f739069f06c776c07b0158af753d642ff8`，含 4653 个张量，不适用于任意其他检查点。

```sh
python tools/prepare-mtp-view.py --model-root /data/Qwen3.8-Flash-Next-W3W4A16-SSD
```

工具检查原始草稿分片，建立独立 `runtime-mtp/`，补充 W4/G32 配方及层号映射；不重新量化或覆盖 no-MTP 权重。启用 MTP 的引擎使用该视图和 `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`，并安装上文的两个输出头补丁。

实际试跑另有独立的 85°C 温度监控和运行时限；一次长代码负载已触发停止。该监控尚未封装为仓库服务入口，不应把短测记录当作散热或生产部署验收。

## 新的短测候选与阶段计时

2026-10-19/20更新：保留chunk128，记录可扩展分配器短测候选；chunk192无收益、
chunk256未通过容量测试。每种输入3次、64输出的结果与OOM记录见
[分块扫描报告](chunk-sweep-20261019.md)。新增[V2 runner阶段诊断](runner-profile.md)
覆盖实际prefill/decode分类、采样、MTP及通信消息大小。

原chunk64基线保留。2026-10-14已短测chunk128候选：其余2K/MTP3/KV256MiB设置不变，
只把 `--max-num-batched-tokens` 改为128。855-token输入的8-output暖态TTFT为1.428s，
匹配chunk64确认是2.445s；样本数分别2/1，不代表持续或通用提速。先预热目标形状，
首次新形状可能触发编译。[完整阶段计时、温度边界和复现方法](stage-prefill-20261014.md)。


2026-10-20 HC小批量融合已完成720组模块数值检查与同进程整模A/B，但没有稳定速度收益；
`Q38_HC_SMALL_BATCH`保持默认关闭。代码与复现工具仅用于继续实验，见
[HC融合报告](hc-small-batch-20261020.md)。
