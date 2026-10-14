# Qwen3.8-Flash-Next-vllm-sm75

面向 **4×Tesla T10 16GiB（SM75）** 的 Qwen3.8-Flash-Next 实验适配：混合 W3/W4 权重、SSD PLE、TP4+EP4、NCCL P2P 和 MTP。实际试跑使用 NUC9、8GB RAM、本地 NVMe 和 16GiB swap，采用原生隔离 venv。

Fork 自[八卡部署仓库](https://github.com/lichangfeng-1/vllm-sm75-8xT10-Qwen3.8-Flash-Next)，引擎固定为 [fishensw/VLLM-SM75 v0.1.7-beta2](https://github.com/fishensw/VLLM-SM75/tree/d7c5d4204cc582bb9c05ff9b4e7a199ae3fbdb0c)。版本锁见 [config/source-lock.json](config/source-lock.json)，上游说明见 [docs/upstream-8xt10-README.md](docs/upstream-8xt10-README.md)。

## 权重方案

- 主模型专家层 2–45：W3/G128；层 0、1、46、47：W4/G128。
- 兼容的 attention、共享专家、GDN、PLE 投影及输出头：W4/G32；Router、norm、HC 等关键控制参数保留原始精度。完整配方见 [quant/recipe.json](quant/recipe.json)。
- no-MTP 主权重约 **50.77GiB**，模型加载约 **13.57GiB/卡**。BF16 PLE 约 **95.37GiB**，放本地 SSD、按需读取，每 rank 显式缓存 32MiB；OS page cache 和查询缓冲另计。
- 可选 MTP 使用原导出的 W4/G32 草稿权重，单独生成运行视图，保留 no-MTP 目录。

量化已在双 CMP 170HX 上完成 48 层校准和导出；源覆盖、保留参数逐字节一致性及文件 SHA-256 检查通过。仓库不包含模型权重。

## 已测结果与限制

2026-10-14 原生四卡简测：FP16 激活、FP8 KV、eager、单并发、2K 上下文、prefill chunk64，每个输入只采样一次并生成 64 tokens。

| 配置 | 439 / 855 tokens 的 prefill | decode | 备注 |
| --- | --- | --- | --- |
| P2P 关闭、无 MTP | 247 / 241 tok/s | 约 7 tok/s | 基线 |
| P2P 开启、无 MTP | 380 / 374 tok/s | 约 7 tok/s | 日志确认 P2P/CUMEM |
| P2P + MTP3 | 350 / 350 tok/s | 14–15.5 tok/s | 接受率约 48–54% |
| P2P + MTP6 | 321 / 322 tok/s | 13.6–15.6 tok/s | 未表现出稳定收益，当前选择 MTP3 |

原始简测数据见 [docs/benchmarks](docs/benchmarks)，逐轮环境和故障记录见 [docs/sm75-trial-20261014.md](docs/sm75-trial-20261014.md)。各轮缓存、温度及 KV 预算存在差异，以上不是多轮统计或严格的单变量性能结论。

**尚未完成 held-out 质量、32K 上下文和持续负载验收。** 长代码生成出现重复输出；一次长负载触发独立监控的 85°C 停止线。短测成功不能视为可长期稳定服务，量化对质量的影响也尚未量化。

## 使用与开发

原生环境、必要补丁与 MTP 视图准备见 [docs/native-trial.md](docs/native-trial.md)。目前提供手动接入工具，尚未提供经过验收的零环境原生安装器或一键服务入口。

```sh
# CPU 配方、SSD 查表及适配逻辑检查
python3 -m unittest discover -s tests -v

# 只读检查目标机器
ssh nuc9 python3 - < tools/preflight-4xt10.py

# 从完整 BF16 源提取 PLE；output 必须是新目录
python3 tools/prepare-ssd-ple.py --source /data/Qwen3.8-original --output /data/q38-ple-bf16

# 接入已完成的兼容量化导出；此命令不执行量化
python3 tools/attach-ssd-ple.py --model /data/q38-w3w4 --manifest /data/q38-ple-bf16/ple-manifest.json
```

PLE 接入采用相对链接，迁移时保留相对目录结构。量化流程见 [docs/quant-job-20261012.md](docs/quant-job-20261012.md)，设计预算见 [docs/4xt10-ssd-plan.md](docs/4xt10-ssd-plan.md)。固定版本 overlay 遇到未知源文件或重复安装时会终止。

`docker/Dockerfile.4xt10-ssd`、`tools/build-4xt10-image.py`、`run/4xt10.py` 和 32K profile 是早期候选方案，默认仅打印命令；其 6GiB 容器内存、禁用 swap、32K 等配置未通过整体验收，也未包含全部原生试跑修复。实际成功试跑使用放开内存限制和 swap 的原生环境。

保留上游 LICENSE 和 NOTICE；旧 `SHA256SUMS.txt` 只对应原八卡版本，不覆盖本仓库新增文件。
