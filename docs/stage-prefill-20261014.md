# 四张 T10：阶段埋点与 chunk128 候选（2026-10-14）

本轮只改引擎参数 `max_num_batched_tokens: 64 → 128`。保持 TP4+EP4、MTP3、
FP16 激活、FP8 KV 256MiB/rank、2K 上下文、seq1、P2P、eager、SSD PLE 和权重不变。
NUC9 四卡均为 100W；没有改驱动、硬件、量化精度或开启挖矿。

## 匹配 HTTP 短测

冻结真实文本输入 855 tokens，每次强制输出 **8 tokens**，temperature=0、seed=42、
no-thinking、prefix caching 关闭。计时请求不启用 CUDA 事件；双方仍保留少量相同的
Python 探针包装调用。模型加载不计入时间。

| 指标 | chunk64 确认 | chunk128 暖态候选 |
|---|---:|---:|
| 正式样本数 | 1 | 2 |
| TTFT | 2.445s | 1.428s（中位数） |
| 服务端 prefill | 2.431s | 1.412s（中位数） |
| 输入 / prefill 中位数 | 351.69 tok/s | 605.53 tok/s |
| 请求总耗时 | 2.759s | 1.768s（中位数） |
| 服务端 decode 区间 | 0.315s | 0.341s（中位数） |

观测 TTFT 缩短 **41.59%**、prefill 缩短 **41.92%**、prefill 吞吐增加 **72.18%**。
候选两次 TTFT 为 1.455926 / 1.400278s；回退 chunk64 后测得 2.444968s。
8-token 输出 SHA256 一致，但只是共同的短回答开头，不能证明完整回答或质量一致。
Decode 区间并未改善；8-token 输出也不足以重新评估稳定 decode 吞吐。

候选 439-token 的两个暖态请求 TTFT 为 0.730916 / 0.715852s，prefill 为
0.722230 / 0.711414s；没有完成该长度的匹配 chunk64/8-output 复测，因此不计算其
前后提升比例。

**冷形状开销没有隐藏：** chunk128 第一次正式 439-token 请求 TTFT 为 **4.794s**，
日志记录 `_qsa_sparse_paged_gqa_splitk_kernel` 等推理期间 Triton 编译。该请求保留在
raw 数据中，单列为冷形状样本，暖态统计不包含它；随后补测一次同长度请求。
部署和测量前应预热目标长度及分块形状。不能把暖态 1.428s 当作首次请求保证。

## 诊断阶段分布

通过 opt-in Python scopes，记录 CUDA events 和 host dispatch wall time。
目标请求仍为 855-token 输入；chunk64 诊断输出16 tokens、chunk128诊断输出8 tokens。
下表仅统计主模型 `tokens>=8` 的 forward：两种配置在每张卡都恰好覆盖855输入行，
所以这部分是可对照的 prefill 范围。**诊断请求不混入 HTTP 性能统计。**

GPU 0 的 inclusive stream 区间：

| 范围 | chunk64 | chunk128 |
|---|---:|---:|
| 主模型 forward 总范围 | 2340.13ms | 1405.27ms |
| MoE/MLP，含内部通信 | 921.05ms | 566.26ms |
| GDN attention | 587.85ms | 350.82ms |
| QSA attention | 572.65ms | 298.17ms |
| HC mix/combine | 177.82ms | 127.51ms |
| PLE，含 SSD embedding | 36.82ms | 34.76ms |
| SSD embedding，含同步/H2D | 29.63ms | 30.35ms |
| all-reduce（嵌套在上面各范围内） | 1068.55ms | 502.43ms |
| 主模型 prefill forward 次数 | 14 | 7 |
| 主模型 prefill all-reduce 次数 | 1358 | 679 |

其他卡也记录了14→7次主模型调用与855输入行。原 chunk64 的 all-reduce 区间跨卡为
约383–1087ms，说明等待和提交差异明显；不能把某张卡的区间全当作 PCIe 纯传输时间。
加大分块减少调用及提交次数，不代表相同输入的通信字节量减半。

这些区间包含提交间隙、等待和嵌套操作；all-reduce 已包含在 MoE/attention 中，
SSD 已包含在 PLE 中，**不能直接相加，也不能把四卡求和当作请求壁时**。
埋点还分别记录 target/draft 的小形状调用；MTP小形状可能出现在prefill辅助计算中，
因此通用汇总只用形状分桶，不直接把 `tokens<8` 标成纯 decode。
输出头、tokenizer、排队等仍以请求指标/残差观察，尚无逐 kernel 的独立拆分。

这次证据支持优先减少细碎分块调用；SSD PLE区间约30ms，未显示它是主要瓶颈。
该负载重复相同文本、缓存较热，不能据此推断冷SSD或更多样输入的读取成本。

## 温度、样本和适用范围

顺序为：chunk64/64-output 基线各长度3次 → chunk128/8-output → chunk64/8-output确认。
最初6次64-output结果只作为补充，不能与8-output做同输出预算比较。
原基线前四次起测上限65°C，后两次70°C；其最初四条结果中的温度字段记录的是请求后
温度，发布副本已更名为 `temperature_after_request_max_c`，原始证据保留。

整模驻留时0号卡空闲仍逐渐变热。候选短测起测上限80°C，匹配长输入两次起测均80°C，
chunk64确认起测81°C。原计划的完整反向重复矩阵因散热余量缩减，不能声称严格的
多轮单变量A/B、统计显著性或通用提升。候选439-token补测起测81°C，单独注明。

三次加载/测试采样最高分别为76/83/84°C，独立2秒轮询、85°C停止线保持不变；
最大采样间隔分别为2.334/2.513/2.244s。没有thermal-stop、掉卡或主机重启。
候选试跑RAM峰值约5.50GiB、swap约7.65GiB。所有本轮模型和监控已停止，GPU显存恢复
4MiB/卡；没有开启常驻服务或挖矿。短测不等于持续散热、质量或32K验收。

## 复用埋点与候选参数

本仓库 runtime 在 `PYTHONPATH`，固定引擎和原生补丁已安装时：

```sh
export Q38_STAGE_PROFILE_CONTROL=/data/q38-profile/control.json
export Q38_STAGE_PROFILE_DIR=/data/q38-profile/stages
# 控制文件不存在时，CUDA events关闭；在启动API/worker前设置上述环境变量。
# 模型就绪、无其他请求时，创建控制文件，再发起一条短诊断请求。
mkdir -p /data/q38-profile
printf '%s\n' '{"enabled":true,"tag":"one-short-request"}' > "$Q38_STAGE_PROFILE_CONTROL"
# 请求完全结束后关闭采集，再做正式HTTP测速。
rm "$Q38_STAGE_PROFILE_CONTROL"
python tools/summarize-stage-profile.py /data/q38-profile/stages
```

`runtime/q38ssd/stage_profile.py` 仅适配固定vLLM0.30.0的eager单请求路径；
不会自动切换精度或修改权重。[候选参数记录](../config/native-2k-mtp3-chunk128.json)
不由现有启动器自动读取。沿用原生2K MTP3命令，仅把
`--max-num-batched-tokens 64` 改为 `128`；其余设置保持原样。
回退改回64。32K Docker候选入口未改，也未因此获得验收。

[发布数据](benchmarks/stage-prefill-20261014/)包含匹配原始计时、保留的冷形状样本、
64-output补充计时和四卡inclusive scope汇总。本机完整原始JSONL、server日志和脚本在
`.artifacts/q38-stage-opt-20261014`、`q38-stage128-opt-20261014`、
`q38-stage64-confirm-20261014`，NUC9有同名目录。
