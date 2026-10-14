# 四张 T10 与 SSD PLE 方案（2026-10-12）

先建立质量可验收、内存有上限的单请求基线，再比较真实负载 prefill、TTFT、decode。PLE 固定放本地 SSD；原八卡包 CPU 全表模式的百 GB 内存要求不适用于此方案。

## 实际硬件

| 项目 | 查实信息 | 影响 |
|---|---|---|
| 主机 | NUC9，i7-9850H，6 核 / 12 线程 | 四卡目标；不是 Z4 双 170HX |
| GPU | 4×T10，每卡 16,384MiB，SM75，当前 100W | FP16 激活；可分配显存小于合计 64GiB |
| RAM | 8,051,970,048 bytes，约 7.50GiB | 首轮容器限 6GiB，实测加载峰值 |
| SSD | 根分区本地 NVMe，检查时约 403GB 可用 | 约 102.4GB BF16 PLE 加普通量化权重 |
| 互联 | Thunderbolt3 + PEX880xx，共用主机上行 | 先禁用未验证 P2P；不引用其他主机结果 |
| 软件 | 可枚举四卡；Docker、nvcc 尚未安装 | 镜像和四卡推理均未运行 |

重启恢复四卡后还没有负载稳定性验证，近期有掉卡记录。首次短测记录温度、Xid、链路；若再掉卡就中止负载，不自动重启。挖矿保持关闭。

## PLE 与主机内存

原始 PLE 约 51.2B 参数，BF16 文件约 102.4GB / 95.37GiB。每个 worker 只读当次请求的完整行，各有 32MiB 显式缓存，四个合计 128MiB，第一版无需 PLE all-reduce。保留 SSD 原始 BF16 字节，进入 SM75 的 FP16 计算时仍有 dtype 转换。

导出器只读 header，用 8MiB 缓冲复制数据；原始分片边界混有普通权重，必须另存专用 PLE 文件。加载专用文件只产生 meta tensor，后端检查形状并跳过整表拷贝；普通权重沿用上游 EP 过滤。现有原生引擎逐行 FP8 scale 不能直接冒充此引擎的全局 FP8 scale。

OS page cache 可回收，但不受 Python LRU 上限控制，会计入 Linux 容器内存。当次查表缓冲随 prefill chunk 增长。第一版无整表预读、无整表 pin。

**PLE 无需百 GB RAM；8GB 整机尚待验证。** 四个 PyTorch/vLLM worker、模型结构、repack 临时张量和 NCCL 都计入峰值。记录 cgroup `memory.current/peak/stat`、anon/file 和进程 PSS，避免把共享库 RSS 简单相加。若触发 6GiB 上限，先分清加载、repack 或 file cache，再决定优化还是增配内存。

## 量化目标

| 参数 | 第一版 | 待验证项 |
|---|---|---|
| 第 2–45 层 routed experts | 对称 W3、G128、FP16 scales | 裸 3bit 加 scales 约 3.125bpw，区别于有效 3bpw |
| 第 0、1、46、47 层 routed experts | W4、G128 | 首尾精度 |
| attention/GDN、shared experts、PLE projection、lm_head | W4、G32 | shared down 的 640/TP4=160 不整除 G128 |
| token embedding | FP16 起步 | 不假定 INC 支持 W4 词表 embedding |
| Router、norm、HC、卷积、控制参数 | 不重新量化 | 单独审计运行时浮点转换 |
| PLE table | BF16 SSD | 每 rank 32MiB 缓存 |
| MTP | 首轮关闭 | 先验收主模型容量和质量 |

必须从完整原始权重重新校准。候选 packing 是 AutoRound INC/Humming；GGUF IQ2/IQ3 不能直接作为此引擎 checkpoint。Humming 官方声明支持 SM75+ 的 INT1–8，但需验证本模型 INT3 MoE、2560/640 维度及 G128；声明支持不等于整模型已跑通。INC mixed-bit 识别要覆盖实际 `RoutedExperts`，不能只匹配旧 FusedMoE 名称。

`quant/recipe.json` 是目标合同，尚非量化执行器。先锁定 AutoRound API/版本并用单层导出验证，再做全量。校准候选起点 128×512 tokens、50 iterations、seed=42，覆盖中英、代码、数学、多轮；记录数据集版本、样本 ID 和每层专家命中数。另留不重叠验收集，对比 BF16、现有 W4、候选，不能用其他 GGUF 的评分代替本权重评分。

## 显存和上下文

2026-10-14 实际 no-MTP 导出：主权重文件 54,515,959,240 bytes（50.772 GiB）；含 tokenizer/config/index 等全部候选文件 54,576,798,910 bytes。这是保留 token embedding、部分投影使用 G32 的实际文件大小，取代此前 49–52GiB 区间估算。BF16 PLE 102,400,491,520 bytes 另计。

文件大小不等于上卡占用。单份约 1.35GiB 的控制参数若 TP4 复制，会额外增加约 4GiB 聚合占用；还要计算 repack、padding、KV、GDN state 和 workspace。按 parameter storage 去重统计每卡实占，不能只把文件大小除以四。

512 个 routed experts 做 EP，每卡 128；普通线性/attention/GDN 做 TP4。QSA 两个 KV heads 可能部分复制，不能假定四份全量 KV。首轮目标 32K、seq=1、prefill chunk=512、每卡 KV 上限 512MiB、FP8 KV、完整 QSA、无 MTP。KV 容量与状态由实际分配验证，32K 是测试目标。

同步 SSD lookup 需要 GPU→CPU 索引，第一版 eager-only，会增加 I/O 延迟。若 PLE 成为瓶颈，再做批量行去重/排序、异步 pread、固定 pinned staging 和有界预取，最后处理 CUDA Graph 边界。性能对比保留完整 QSA，不套用原实验 TP4 的关闭 indexer/CPU 冷专家模式。

## 验收顺序与当前状态

1. 固定版本，验证 SM75 W3 MoE、mixed-bit 元数据、TP4 维度和 SSD GPU adapter。CPU、SM80 小模型校准/packing/streaming 对照和 Transformers SSD lookup 测试已通过；SM75 推理镜像尚未构建验证。
2. 官方 BF16 已补齐，131/131 分片按固定源 SHA-256 校验通过；原始 BF16 PLE 已分离成 128 个专用 SSD 分片，约 102.4GB。原 MLX-4bit 经用户明确授权删除。Z4 双 170HX 已完成全部 48 层 W3/W4 量化；no-MTP 候选通过 CPU 完整性和 SHA-256 验收，具体记录见 `docs/quant-job-20261012.md`；原始源保留。
3. 在 Z4 等量化主机分层校准导出，NUC9 不承担全模型校准。现有 ComfyUI 不自动停止。量化产物做格式/数值/held-out 验收。
4. 将普通权重和 PLE 放到 NUC9 本地 SSD，接入 index；先 eager + NCCL 回退，测 6GiB 容器上限内的启动峰值及每卡 repack 后显存。
5. 先 4K，再 16K/32K；记录温度、Xid、内存。通过后验证四卡 P2P payload/all-reduce 正确性和性能，再选择通信路径。
6. 相同输入、实际 token 数、生成长度、缓存冷热及采样设置下比较。记录 PLE、QSA/attention、GDN、MoE、通信耗时，以及 TTFT p50/p95、decode tok/s。prefix cache 命中不能当作冷 prefill 提速。

重新量化和导出 CPU 验收已完成。GPU 集成、质量、8GB 实测或 32K 容量仍未验证；按用户要求，本轮没有试跑完整模型。

来源：[八卡基准包](https://github.com/lichangfeng-1/vllm-sm75-8xT10-Qwen3.8-Flash-Next)、[固定 SM75 引擎](https://github.com/fishensw/VLLM-SM75/tree/d7c5d4204cc582bb9c05ff9b4e7a199ae3fbdb0c)、[Humming](https://github.com/vllm-project/humming)、[AutoRound](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md)。硬件来自本轮只读查询。
