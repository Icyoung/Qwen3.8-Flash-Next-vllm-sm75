# 本轮验证记录

- `python3 -m unittest discover -s tests -v`：11 项通过。覆盖 BF16 原始字节提取、混合源分片、跨页/重复/乱序查表、缓存上限、空输入、越界、损坏文件、header 校验、路径限制、index 备份、普通权重 EP 参数传递，以及 PLE iterator 只请求 meta allocation 的契约。
- meta allocation 测试使用轻量 spy，不是实际 PyTorch/GPU 加载测试。
- 两处 SSD overlay 对固定上游文件的 hash 验证和补丁生成成功，生成 Python 可解析；未安装到运行中的 vLLM。
- 新 Python 文件语法、JSON、TP4 参数、eager 参数、禁用未验证 PCIe IPC，以及容器内存/swap 限制检查通过。
- 固定版本构建命令 dry-run 通过；未运行 Docker build。
- NUC9 只读 preflight：四卡均为 Tesla T10 / SM75 / 16,384MiB，driver 595.91.07，100W，空闲温度 42–47°C；RAM 8,051,970,048 bytes，root free 432,076,570,624 bytes。Docker、nvcc 均未安装。

尚未验证：INT3 SM75 dense/MoE、完整混合位宽导出、GPU SSD adapter、全模型上卡容量、8GB 主机加载峰值、32K 上下文、重启后的负载稳定性、NUC9 P2P、真实 prefill/TTFT/decode。没有新的性能数据。
