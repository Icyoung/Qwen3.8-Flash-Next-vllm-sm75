# SM75 HC 小批量融合与四卡验证

结论：**整模没有稳定收益，保持默认关闭。** 保留实验开关、复现工具和失败证据，
现行chunk128/P2P/MTP3/可扩展分配器候选不启用这条HC路径。

本轮针对前一轮 decode 诊断中的 HC 路径：MTP3 主模型通常一次验证4个token，
草稿模型主要为单token。原引擎的可选 `VLLM_SM75_QWEN38_HC_GEMV` 只覆盖单行，
本轮仍保持该开关为0，单独验证新的融合路径。

## 实现范围

`Q38_HC_SMALL_BATCH=1` 将 HC 的上投影、sigmoid门控、四路加权平均合并为一个
Triton kernel。FP16权重和输入不变，乘加及门控累加使用FP32，并保留原上投影输出
的FP16舍入边界。没有修改检查点、量化位宽、norm、下投影/注入、SiLU或残差combine。
也没有复制常驻权重，省去的是上投影中间gate张量及一次消费者kernel调用。

仅适配固定vLLM0.30.0和引擎overlay中的HC源文件，安装时校验两个文件SHA-256。
仅SM75、eager、FP16、HC=4、hidden=2560、rank=320、1–4行连续张量进入新路径；
较大批量和其他布局走原函数。安装时拒绝非eager配置；不声称支持CUDA Graph或compile。

仅为复现实验，在固定原生环境、runtime已加入PYTHONPATH、chunk128候选设置之上启用：

```sh
export Q38_HC_SMALL_BATCH=1
export VLLM_SM75_QWEN38_HC_GEMV=0
```

回退：unset `Q38_HC_SMALL_BATCH` 后重启模型进程。关闭时不安装HC包装或新kernel。

## 数值与模块验证

四张T10分别使用检查点层0、24、47、最后mixer及MTP的真实HC权重；随机输入尺度
0.1/1/3，行数1/2/3/4/5/128，mix与combine_and_mix两种调用，共720组全部通过。
新旧block输出最大相对L2为0.000045964（约0.0046%），最大绝对差0.00390625。
残差与注入输出逐元素一致，5/128行回退输出逐元素一致。这是模块数值验证，不能替代
held-out模型质量评测。

可复现工具需在空闲且有独立温度监控的GPU上运行；不加载完整模型：

```sh
PYTHONPATH="$PWD/runtime" python tools/validate-hc-small-batch.py \
  --model /data/Qwen3.8-Flash-Next-W3W4A16-SSD/runtime-mtp \
  --output /data/hc-validation.json --devices 0,1,2,3
```

首轮GPU0 eager微测试，CUDA事件包住重复Python调用，包含提交间隙：
上投影+门控单行约67→26µs，4行约68→46µs。实际整个HC mix路径单行约228→208µs，
4行约234→215µs，约8%的模块下降。128行回退测试约231→253µs，有包装检查开销；
这些是暖权重小模块测试，不应直接换算整模收益。正式A/B需要同时核对prefill是否回退。

[数值与微测试数据](benchmarks/hc-small-batch-20261020/)保留原始候选tile扫描，
扫描中的25.88µs最低点不是最终选择；实际代码单行ROWS16/4warps，多行ROWS8/4warps。

## 同进程整模 A/B

2026-10-20 NUC9，125W/卡，TP4+EP4、MTP3、P2P/CUMEM、chunk128、FP16、
FP8 KV256MiB/rank、2K、seq1、SSD PLE、eager、expandable_segments。
挖矿关闭。模型只加载一次，A/B各预热hello/439/855三组8-token请求，随后按
A-B-B-A-A-B顺序运行，每块各439/855两种输入、每次64输出。每个配置/输入共3次。
固定temperature0、seed42、ignore_eos、prefix cache关闭。每种长度的出场先后交替。

A/B切换仅用于实验：各worker在runner execute边界读取同一原子更新控制文件，
记录模式和融合调用数；两种模式都有该控制包装。四worker均确认B块实际调用，
正式B块每卡累计1-row1176次、4-row16300次；A块为0。发布的静态开关没有文件轮询。
模型和探针源码不在测试中热替换；正式测速关闭CUDA events，两次诊断单独保存。

| 输入 | 路径 | TTFT中位数 | Prefill中位数 | Decode中位数 | Decode范围 |
| --- | --- | ---: | ---: | ---: | ---: |
| 439 | 原路径A | 0.693s | 643tok/s | 15.48tok/s | 14.77–15.61 |
| 439 | 融合B | 0.691s | 639tok/s | 15.21tok/s | 15.17–15.75 |
| 855 | 原路径A | 1.208s | 711tok/s | 14.22tok/s | 13.75–16.21 |
| 855 | 融合B | 1.218s | 705tok/s | 14.30tok/s | 13.38–14.73 |

各指标独立中位数。中输入decode约−1.8%，长输入约+0.5%，不足以采用。
输出哈希和MTP接受率仍变化：439输入A/B接受率51.32%/50.65%，855为47.68%/44.31%。
此处不能把接受率差异定性为模型质量下降，也不能用不同生成轨迹的单次总耗时
证明kernel收益。每组仅3次，未做长期稳定性或held-out质量评测。

单独的855→64诊断，GPU0按各自decode execute次数归一化：

| 指标 | A | B |
| --- | ---: | ---: |
| Decode execute次数 | 28 | 26 |
| 每execute的runner CUDA区间 | 167.06ms | 165.87ms |
| 每次HC调用的CUDA区间 | 92.59µs | 106.57µs |
| 每次HC调用的host dispatch | 345.96µs | 320.98µs |

融合减少host dispatch，但该诊断中的HC CUDA区间增加。runner每步区间只降约0.7%，
正式HTTP没有稳定收益。这说明暖权重微测试的提交开销减少，不能直接换算成整模提速。
CUDA区间为inclusive events，嵌套范围不可相加，且诊断同步会扰动执行，不能将它
称为纯kernel活跃时间。原始四卡汇总一并保留。

整轮监控最高45°C，最大采样间隔2.26s，NVML显存最高15833MiB/卡（含诊断），
RAM峰值5588635648B、swap峰值8151588864B；85°C guard未触发。
模型与测量unit正常结束，四卡恢复4MiB/0%，挖矿仍关闭，未重启主机或改动功耗/驱动。

下一步优先检查MoE与GDN的小批量路径。当前固定引擎的GDN decode日志明确走Triton，
其另一个fused CUDA分支要求SM80+及BF16等条件；不能直接通过开关让该分支运行在SM75。
本轮没有修改GDN或MoE计算。
