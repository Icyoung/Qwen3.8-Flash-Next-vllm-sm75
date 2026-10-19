# 四 T10：分块扫描、显存边界与 decode 诊断

2026-10-19 晚至 10-20 凌晨在 NUC9 实测。结论：**保留 chunk128**；
chunk192 没有显示收益，chunk256 未通过完整短测。为 chunk128 记录
`PYTORCH_ALLOC_CONF=expandable_segments:True` 候选，采样显存峰值有所下降。
该参数改善后的短测不代表长期稳定或任意形状均不会 OOM。

## 条件与正式结果

四卡均 125W，挖矿关闭；TP4+EP4、FP16、原混合 W3/W4 权重、SSD PLE、MTP3、
FP8 KV 256MiB/rank、2K、seq1、eager、NCCL P2P/CUMEM。没有重新量化、减少 KV
预算或更改驱动。正式 HTTP 请求不启用 CUDA events，预热和诊断另存。
固定提示词、temperature=0、seed=42、ignore_eos、每次生成64 tokens。

| 配置 | 每种输入的样本数 | 439 TTFT | 855 TTFT | 439 prefill | 855 prefill | 439 decode | 855 decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk128，原分配器复测 | 2 | 0.706s | 1.326s | 628 tok/s | 649 tok/s | 14.66 tok/s | 14.19 tok/s |
| chunk192，原分配器 | 3 | 0.743s | 1.420s | 598 tok/s | 604 tok/s | 14.25 tok/s | 13.08 tok/s |
| chunk128，可扩展分配器 | 3 | **0.681s** | **1.253s** | **648 tok/s** | **686 tok/s** | 15.37 tok/s | 13.77 tok/s |

均为各指标独立中位数。Decode 用63除以服务端 decode 区间，不用 SSE 消息条数。
可扩展分配器下的855-token TTFT范围1.251–1.273s；decode范围13.66–16.10tok/s。

**不能把差值全归因于分配器：** 顺序测试，缓存历史、加载次数和起始温度不同；
原chunk128未挂runner探针，chunk192仅model探针生效，最后chunk128启用了V2探针
包装但正式请求关闭采集。完整输出的哈希存在差异，decode的草稿接受情况也变化。
因此这里只报告观察值，不能声称3–6%的因果提速或一致的decode提升。

## chunk256为何没有采用

1. 原分配器：hello成功，439-token预热在PLE gate分配20MiB临时张量时OOM。
   报错时仅7.56MiB可用，另有214.53MiB为PyTorch保留但未分配；最高采样温度41°C。
2. 加入可扩展分配器：hello及439-token预热通过，后者TTFT0.570s；随后855-token
   预热在QSA sparse paged GQA split-K kernel启动时出现Triton CUDA OOM，最高42°C。
   没有正式测速样本，不能把0.570s列为可采用配置的成绩。

第一次失败提示碎片可能参与，第二次结果说明单改分配器仍不足以通过这组负载。
后续若继续增大chunk，应先分析PLE/QSA临时张量和kernel启动所需显存，而不是根据
模型刚加载完的空闲显存判断容量。错误记录及两次预热结果均保留。

## V2 runner诊断

新增的`Q38_STAGE_PROFILE_RUNNER=1`适配实际使用的V2 runner。第一次chunk192试验
的新增钩子误对齐到V1，因此其trace只有原model scopes，phase为unknown；该轮不用于
完整decode分布结论。修正后，在最后chunk128试验完成GPU验证。

诊断单独使用855-token输入、64-token输出；四张卡均验证了855行prefill、7个prefill
execute、26个decode execute及26个decode sample。零token清理步骤另记unknown，
不计入decode。下表为GPU0，单位ms：

| 范围 | Prefill | Decode |
| --- | ---: | ---: |
| runner execute | 1252.26 | 4214.77 |
| 主模型 forward，包含下面各层及内部通信 | 1226.06 | 4117.01 |
| MoE/MLP，含内部通信 | 492.52 | 1664.47 |
| GDN attention | 296.15 | 959.65 |
| HC mix/combine | 106.25 | 912.82 |
| QSA attention | 277.08 | 381.05 |
| SSD PLE读取/同步/H2D | 29.57 | 31.11 |
| runner sample，含输出头/采样/草稿/状态处理 | 122.92 | 464.92 |
| MTP草稿提议，含草稿模型及输出头 | 113.10 | 424.90 |
| 主模型输出头及采样/拒绝 | 5.70 | 31.58 |

范围互相嵌套，不能把各行相加；CUDA事件包括提交间隙和等待，不是纯kernel活跃时间。
诊断会增加event开销并在execute/sample边界同步，不能把它当作正式HTTP耗时。
这些数据支持下一步优先检查MoE/MLP、GDN及HC小形状路径；SSD PLE在这组暖缓存重复
文本里不是主要耗时。它不证明冷SSD、多样文本或长上下文也如此。

主模型prefill每卡679次all-reduce：582次655360字节、97次445440字节；decode每卡
2522次，每次20480字节。GPU0对应区间415.61/348.16ms，包含在上面各层中。
这是collective输入张量大小，不是PCIe实际传输字节数。MTP和输出头另有通信。
chunk192虽然将prefill降到5次forward/485次all-reduce，GPU0主模型区间仍为1444.74ms，
说明减少调用次数本身不保证更快。

完整使用方法及计时边界见[runner诊断说明](runner-profile.md)。

## 显存、监控与复现

| 试验 | 显存采样峰值/卡 | 最高温度 |
| --- | ---: | ---: |
| 原chunk128复测 | 15909MiB | 52°C |
| chunk192 | 15901MiB | 45°C |
| chunk256失败 | 15929MiB | 41°C |
| chunk256可扩展分配器失败 | 15839MiB | 42°C |
| chunk128可扩展分配器 | **15673MiB** | **45°C** |

采样约每2秒一次，可能漏掉瞬时峰值；不能用该表的空闲量推翻OOM报错。
最后chunk128相对最初复测的采样峰值低236MiB/卡。独立85°C guard未触发，未观察到
主机重启或掉卡。所有试验结束后模型和监控退出，GPU恢复4MiB，挖矿保持关闭。

候选参数见[chunk128配置](../config/native-2k-mtp3-chunk128.json)。现有启动器不会
自动读取该文件；沿用固定原生启动参数，chunk保持128，额外设置：

```sh
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

回退时unset该变量，保留chunk128。不要将未通过的chunk256设为默认。
held-out质量、持续负载和长上下文仍未验收。

[发布数据](benchmarks/chunk-sweep-20261019/)包括原始正式/预热计时、失败状态、
四卡stage汇总和监控统计。完整日志、JSONL、控制器及监控脚本保留在本机
`.artifacts/q38-*-20261019`及NUC9同名试跑目录中。
