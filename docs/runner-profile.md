# 原生 V2 runner 阶段诊断

固定 vLLM 0.30.0、Qwen4Exp NVIDIA 实现、V2 runner、eager、单请求路径。
原来的 model-only 计时仍可使用；新增 runner 模式覆盖调度步骤中的主模型、输出头、
采样/拒绝、MTP 草稿、输入准备、状态更新和输出复制提交。

## 启用

在 API 和 worker 启动前设置，保持控制文件不存在：

```sh
export PYTHONPATH="$PWD/runtime${PYTHONPATH:+:$PYTHONPATH}"
export Q38_INC_INT3=1
export Q38_STAGE_PROFILE_CONTROL=/data/q38-profile/control.json
export Q38_STAGE_PROFILE_DIR=/data/q38-profile/stages
export Q38_STAGE_PROFILE_RUNNER=1
```

省略 `Q38_STAGE_PROFILE_RUNNER` 则沿用 model-only scopes。没有控制文件或其中
`enabled=false` 时不创建 CUDA events。Python 包装调用和文件存在性检查仍存在；
不要宣称关闭采集后绝对零开销。CPU 测试不能验证 GPU 钩子，实际诊断数据必须包含
四个 device 的 `runner_execute`、`runner_sample` 和 `phase=decode`。

服务已就绪且独立温度监控在运行时，执行：

```sh
python3 tools/benchmark-native.py \
  --url http://127.0.0.1:18440 \
  --model Qwen3.8-Flash-Next-trial \
  --output /data/q38-profile/chunk128-results.json \
  --telemetry /data/q38-profile/telemetry.jsonl \
  --profile-control /data/q38-profile/control.json \
  --require-mtp

python3 tools/summarize-stage-profile.py /data/q38-profile/stages
```

输出目录需预先存在，测速文件必须是新路径。工具先保存三条 8-token 预热结果，
再交替顺序测两个固定提示词，每种三次、生成 64 tokens。正式计时结束后才原子写入
控制文件，另外发一条 64-token 诊断请求，最后移除控制文件。诊断 HTTP 时间单独写入
`.diagnostic.json`，不得混入性能中位数。正式请求的 Prometheus counter 增量必须对应
恰好一个请求，MTP 验证要求草稿计数实际增加。

`--telemetry` 接受试跑独立 monitor 的 JSONL：每行含 Unix `time` 和四个 GPU 的
`temperature_c`。工具拒绝超过 6 秒的旧采样，或在 80°C 以上启动下一个请求。
它不负责请求执行过程中的温度保护，仍需外部 85°C guard、服务运行时限及退出清理。

## 计时语义

- `phase` 使用 V2 `gather_batch_req_state` 返回的 CPU 进度和 prefill 长度。
  一个 token 的 prompt 尾块仍为 prefill；混合批次标为 mixed。旧 trace 为 unknown。
- `runner_execute` 包含输入准备和主模型 forward；`runner_sample` 包含输出头、
  采样、草稿提议、状态处理和复制提交。它们共享 step 编号。
- sample 记录的 `tokens` 是同一步 execute 的调度元数据，不能与 execute 相加当作
  新处理的输入行数。draft 内部调用和 padding 也不能用这个值倒推。
- `target_forward`、`draft_forward`、`draft_proposal_including_models`、
  `sampling_including_logits_and_rejection`、各 attention/MoE/HC scope 均是 inclusive。
  all-reduce 已在这些 scope 内，不能重复求和；四张卡的时间不能求和当请求壁时。
- `input_bytes_histogram` 统计 collective 的输入张量字节数及调用次数，不是 PCIe
  实际线上传输量，也不包含 NCCL 的算法流量放大。
- CUDA event 区间包含 stream 等待及 CPU 提交间隙，并非纯 kernel 活跃时间。
  `root_host_dispatch_ms` 是原方法返回前的 CPU 壁时，含其内部同步等待；
  `root_wall_ms` 还含诊断所加的最终同步和汇总开销。
- `output_copy_enqueue` 仅测复制提交，不包含另一个 stream 上的完整复制完成时间。
  调度器/进程间消息、异步 output 的后续消费和 HTTP/网络仍以端到端指标观察，
  不把未覆盖时间直接命名为 CPU 或通信瓶颈。
- 新模式在 execute/sample 边界同步一次，不在每个主模型/草稿子 forward 结束后同步。
  诊断仍会扰动调度；只能定位下一轮优化方向，不能替代关闭采集的 HTTP 测速。

CPU 检查：`python3 -m unittest discover -s tests`。覆盖短 prefill 尾块、decode、
mixed、空批次，以及跨 tag/phase 分组和通信大小直方图合并。
