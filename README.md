# vLLM-SM75 0.1.7 自包含部署包（8 × Tesla T10）

在 **G292-Z20 / 8 × Tesla T10（sm_75，纯 PCIe、无 NVLink）** 上，把
**Qwen3.8-Flash-Next-W4A16-FP8PLE** 这个档（TP8＋EP、256K 上下文、FP8 KV、QSA 全开、无 MTP）
从"能跑"做到"跑满"。对应上游 vLLM-SM75 **0.1.7-beta2**。

每一步都写了**怎么算成功**，不满足就停在那一步。全程照做约 20–40 分钟（大头是模型加载；
换镜像后的引擎冷加载是另一笔账，见 `部署文档-v1.md` 的"切换与回退要排的窗口"）。

**上游项目**：<https://github.com/fishensw/VLLM-SM75>（Apache-2.0）——一个 Turing（sm_75）专用 vLLM 分支，
整合 FlashInfer 0.6.18（PR #3526）与 SM75 后端门槛补丁（PR #47949），持续同步 vLLM 主线。
**本包不是它的 fork，也不含它的源码树**：本包只是这台机（8 × T10）上的部署脚本、一层镜像补丁与一层自维护回填，
构建时要你自己把那份源码树指过去（`SM75_SRC=`）。署名与授权逐条见 `NOTICE`。

---

## 1. 有什么效果

同一台机、同一个档、同一把尺（Ultra 控制台自带「模型测试」，并发 1、超时 30 s），
差别只是镜像多一层 PCIe-IPC 回填 ＋ 档里多一个环境变量：

| 指标 | 装之前 | 装之后 | 变化 |
|---|---|---|---|
| 每 token 步时 ITL | 22.89–23.25 ms | **17.22–17.61 ms** | 每步省 ≈5.5 ms |
| 输出 tok/s | 43.15–43.95 | **57.03–58.36** | **＋30.5%～＋33.1%**（逐档配对） |
| 预填充 tok/s、TTFT | — | — | 1% 以内（噪声） |

实测截图（生产档，九个提示词长度 512→131072，2026-10-07 14:3x）：

![生产档 PCIe-IPC：ITL 17.22–17.61 ms、输出 57.03–58.36 tok/s、预填充 2405.74–3117.93 tok/s](docs/bench/20261007-pcieipc-production.png)

装之前那批（后端 FIREFLY_AR，五个长度）：[`docs/bench/20261007-baseline-firefly-ar.png`](docs/bench/20261007-baseline-firefly-ar.png)

用官方随包尺子复核过 8K 档：decode 44.55 → **58.01**（＋30.2%），`passed=true`、零抢占、零缓存命中、无投机。
完整数据与判读：`docs/实测-面板测速对照-v1.md`。
**这些数字只在"8 × T10 ＋ sm_75 ＋ @150 W ＋ 下面那份权重修订"这一组条件下成立**，换机器不可套用。
还有一条要与版本对齐：**这批数据取在 0.1.7-beta 基座上**（`docs/实测-面板测速对照-v1.md` 自标当时口径），
包 v1.2 换到 beta2 后**未重测**——那层回填的机理不随基座变，但严格说这组数字要在新基座上重跑一次才算闭合。

## 2. 基础条件

| 项 | 要求 |
|---|---|
| GPU | 8 × Tesla T10（sm_75，16 GiB），纯 PCIe 无 NVLink。其它 sm_75 卡需重测常量 |
| 宿主驱动 | ≥ 570；`nvidia-smi` 要能出八张卡。**驱动小版本升级不必默认判红**：580.178.04 与本包的 to20s／pcieipc 两层补丁同机实证可用（引擎起得来、九格跑完、异常计数 0；发布机上叠在上面的两层自维护补丁也同机跑通）。但驱动一动就要复核 P-State 那枚库（见下一行的 ② 与 `docker/NVAPI-获取说明-v1.md`） |
| 容器运行时 | Docker ＋ **NVIDIA Container Toolkit**（`docker info` 的 Runtimes 里要有 `nvidia`，否则 `--gpus all` 起不来） |
| 宿主 CUDA toolkit | **不需要**。CUDA 12.9 与 `nvcc` 都在镜像里（这台机宿主就没有 `/usr/local/cuda`）。注意别和上一行的 Container Toolkit 混——那个是必须的 |
| 内存 · FP8PLE 靶子 | 总 ≥ 128 GiB、可用 ≥ 110 GiB；容器上限 112 GiB（权重加载峰值实测 110.7/112，别调低） |
| 内存 · AutoRound 靶子 | **总 ≥ 240 GiB**（实测容器内存占用 231.4 GiB、宿主 used 153 GiB）。128 GiB 的机器跑不了这一靶，走 FP8PLE |
| 磁盘 | 权重约 120–124 GiB；根盘余量 ≥ 40 GiB |
| 面板两处必修 | ① GPU 探测超时 3 s→20 s（本包的层1 已含）② P-State 库：**默认用宿主驱动自带那份**（`NVAPI=auto` 探三条标准路径）。宿主没有就自动降级成"不带库＋电源档落 sleep"；**缺库而电源档还是 pstate 的后果是引擎根本起不来**（面板那句校验不过就抛，抛在启动引擎之前），不是"少个省电"——见 `docker/NVAPI-获取说明-v1.md` |

### 先把模型文件准备好（两靶子任选其一，下载要挑时间）

**靶子 A · FP8PLE（本包对外数字全部出自这一靶，默认）**
`Qwen3.8-Flash-Next-W4A16-FP8PLE`，与 0.1.6 包跑的是同一份权重谱系。
官方给的取件地址是 Hugging Face 的 `albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`，
**必须用这个修订** `ef554143369a706525336f6b42a09094835dc077`（简写 `ef55414`）——数字对不对全看它。
魔搭 ModelScope 上我们没搜到同名档，所以取件走 HF；国内网络可在下载容器里加 `-e HF_ENDPOINT=https://hf-mirror.com` 换镜像站。
拿到后**先全量校验再往下走**：`bash tools/sha256-weights-v1.sh /你的/权重目录` ⇒ 末行 `SHA_VERIFY_PASS 34/34`。

**靶子 B · AutoRound W4A16（Intel 微调量化档，可选）**
`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`，魔搭可直下：

```bash
modelscope download --model Intel/Qwen3.8-Flash-Next-W4A16-AutoRound --local_dir /你的/权重目录
```

这一靶走包内 `run/profiles/flash-next-tp8-256k-autoround.json`（`format=custom`、`HC_GEMV=0`）。
它的实测吞吐**低于靶子 A**（九长度配对：输出 −2%～−13%，差值集中在 decode 每步），
内存要求却高得多——选它之前先对上表那一行 256 GiB。两靶子的数字不互套。
完整取件与下载命令的另一种写法在上游源码树 docs 目录下的 flash-next-tp8.md（那是源码树自带的文档，不在本包内）。

## 3. 怎么用

```bash
# ① 构建镜像（层0 官方 ultra → 层1 探测超时 20s → 层2 PCIe-IPC 回填）
#    已有官方基座标签 vllm-sm75:v0.1.7-ultra-beta2 就只补层1、层2；
#    没有就指一份 0.1.7-beta2 源码树（本包不随附源码树，取件见下面两行与 部署文档 步骤 2）
bash docker/build.sh
git clone --branch v0.1.7-beta2 --single-branch https://github.com/fishensw/VLLM-SM75.git
SM75_SRC=/path/VLLM-SM75-0.1.7-beta2 bash docker/build.sh    # 需要连层0 一起建时

# ② 机器体检（只读）。有 BLOCK 会以退出码 4 结束，先解决再继续
bash run/env-check-v1.sh /你的/权重目录

# ③ 起控制台容器（先看预演：--dry-run 不写任何东西、不起容器）
DATA=/var/lib/sm75-console MODELS=/var/lib/sm75-models/Flash-Next-FP8PLE bash run/start-here-v2.sh --dry-run
DATA=/var/lib/sm75-console MODELS=/var/lib/sm75-models/Flash-Next-FP8PLE bash run/start-here-v2.sh

# ④ 注册权重＋从官方模板建档
docker exec <容器> node /console-data/tmp/console-driver-v1.cjs setup          # 打印 PROFILE_ID

# ⑤ 问答式配置：几个人用、要多长上下文、CPU KV 开多大、多久进 P8……
#    回车＝基线；先 --dry-run 看一遍将要改什么，它一条都不会写
bash run/configure-v1.sh --dry-run
bash run/configure-v1.sh
```

⑤ 是给人用的，也是给"不想记参数"的人用的：它把取值范围、内存/显存账和联动关系收在一处——
多人用会顺手抬 CPU KV 并提示容器上限要重建才生效；内存不够的机器它直接拦并告诉你最多能开多大；
把上下文往低改会二次确认（那是能力项不是性能项）。PCIe-IPC 那一题**默认开**（这台机上 decode ＋30% 就是它）。
不想用问答也行：`docker exec <容器> node /console-data/tmp/console-edit-profile-v3.cjs <profileId> dump`
先读当前值，再按 `setenv`/`setarg`/`addarg`/`setfield` 逐条改，每条自带回读断言与回滚。

```bash
# ⑥ 起引擎，看签名
docker exec <容器> node /console-data/tmp/console-driver-v1.cjs start <profileId>
docker exec <容器> node /console-data/tmp/console-driver-v1.cjs status <profileId>
```

浏览器开 `http://127.0.0.1:1615` 就是 Ultra 控制台，登录 token：`docker exec <容器> cat /console-data/key`。
⑤⑥ 在网页上点也能做（注册权重 → 选模板 `flash-next-tp8-256k-nomtp` → 改参数 → 启动）。

**怎么算成功**（`start-here` 自己会逐条断言，不过就非零退出）：

- `RUN_DONE 全部终态断言通过`
- 引擎日志 `Using ['FLASHINFER_PCIE_IPC', 'FIREFLY_AR', 'PYNCCL'] … for group 'tp:0'` —— 后端在**第一位**
- 有 `Initialized FlashInfer PCIe IPC all-reduce`，**没有** `does not provide PcieIpcAllReduceWorkspace`
- `GPU KV cache size: 339,110 tokens`、`FP8 layout verified` 恰好 **96 条**、`backend=p2p` 8 条、无 `Traceback`

逐步操作、每步判据、失败与处置对照表：**`部署文档-v1.md`**。

## 4. 包里有什么

```
README.md                本文件
部署文档-v1.md            逐步操作与判据、失败处置表
基线与口径说明-v1.md       哪些值来自官方、哪些是这台机的差异、数字怎么读
CHANGELOG.md              版本沿革
LICENSE  NOTICE  SHA256SUMS.txt
run/    start-here-v2.sh  唯一启动入口（门禁＋终态断言＋读 config.env）
        configure-v1.sh   问答式配置（并发／上下文／CPU KV／进 P8 秒数／单提示图数／端口…，回车＝基线）
        env-check-v1.sh   只读机器体检
        profiles/         两靶档：FP8PLE（默认）与 AutoRound W4A16，PCIe-IPC 都已默认开，
                          单条提示图数写死 image=16/video=0（v1.2 前这一条根本没写、吃引擎默认值）
docker/ build.sh          唯一构建入口（三层，逐层终态断言，任一层可跳）
        Dockerfile.to20s-v1 / to20s-patch-v2.sh      层1：探测超时 3s→20s
        Dockerfile.pcieipc-v1 / pcieipc/             层2：PCIe-IPC 回填物料与门禁
        benchlocal.sh                              「模型测试」三枚受管验证容器的官方启动器（逐字节取自上游 beta2）
        libnvidia-api.so.1 / NVAPI-获取说明-v1.md    P-State 库说明：v1.2 起默认探宿主自带那份，包内这枚是 opt-in
tools/  panel/            面板命令行驱动（建档/起停/改档）。改档工具**以 v3 为准**；v2 作废原因＝clone 的 id
                          字符集比上游删除路由宽（放行 . 与 _ 会建出删不掉的孤档），v1 是更早一版；两份都留在包内但不被引用
        sha256-weights-v1.sh / flash-next-tested.sha256   权重全量校验（34 项）
        gen-console-env-v1.sh   把在役容器 env 固化成宿主 600 文件
        bench-official-v1.sh    官方尺子测速封装（口径钉死＋可比性判定）
        lint-package-v2.sh      我们自己的出厂自检。发布侧注意三条：① 它**必须**带一份仓库之外的身份清单才全绿——
                          `PRIVACY_PATTERNS=/你的路径/privacy-patterns.txt bash tools/lint-package-v2.sh`；
                          **从干净 clone 直接跑 C 段会 FAIL，那是设计如此、不是包坏了**（把姓名/用户名/宿主路径/
                          内网段写进包内文件，等于让检查器自己成为泄露源）。清单格式＝一行一条 ERE、`#` 开头是注释，
                          例如两条通用形状：`^[a-z]+@[a-z.]+\.cn$` 与 `^(10|192\.168)\.`。
                          ② `--fix` 会**整表重写** `SHA256SUMS.txt`，与本包"清单就地增改、不整表重生"的规矩相反
                          ⇒ 只用于自己看形状，发布前请改回就地增改（这样 diff 里看得清动了哪几条）。
                          ③ v2 对"逐字随包的上游文件"有一条**哈希核过才生效**的窄豁免（见 CHANGELOG v1.2）。
                          日常使用者不需要跑它，核对完整性用 sha256sum -c 就够。
docs/   实测-面板测速对照-v1.md ＋ 两张截图 ／ 群友提示词-PCIeIPC回填-v1.md
```

**不含**：模型权重、0.1.7-beta2 源码树、任何凭据。

## 5. 六条要知道的

1. **那层 PCIe-IPC 回填是我们自维护的补丁，不是官方能力**：把 FlashInfer 0.7.0.post1 的 6 个文件与 3 处导出补进镜像里已装的 0.6.18，
   不升级 FlashInfer、不动 attention 内核与 cubin、vLLM 树一个字节不改。官方镜像点不亮这个后端。
   一键退回官方形态：`SKIP_PCIEIPC=1 bash docker/build.sh`（或 ⑤ 那一题选"关"）。上游带上这个能力后**应当撤除**。
2. **改配置走 `run/configure-v1.sh`，别手改 JSON**。它把答案分两处落：档内的（并发、上下文、CPU KV、电源）
   通过面板 API 改并逐字段回读，容器层的（端口、内存上限、绑定）写进 `$DATA/config.env` 由 `start-here-v2.sh` 读。
   容器内存上限这类**要重建容器才生效**，脚本末尾会把那条命令打给你。
3. **端口默认只绑 127.0.0.1，包内不含凭据**。要局域网访问显式 `BIND_HOST=0.0.0.0` 并自行加鉴权；
   登录 token 与引擎 key 都在运行时生成、落在 `$DATA`（`chmod 700`），别外发那个目录、别提交进 git。
4. **两处与官方文档不同**（都写进 `基线与口径说明-v1.md` 的漂移项）：面板数据我们绑到容器内 `/console-data`（官方示例绑 `/data`，
   两者都行，硬要求是 `SM75_CONSOLE_ROOT` 与 bind 目标一致）；编译缓存我们用四条 home 目录 bind
   （`.cache`/`.triton`/`.nv`/`.tilelang`，官方是 `/data/cache` ＋ 一组 `*_CACHE` 环境变量），
   多出来的 `.tilelang` 是实测落在可写层、重建会重付编译费才补的。
5. **P-State 库默认跟宿主驱动走**（v1.2 起）：`NVAPI=auto` 探宿主自带那份，探不到就自动"不带库＋电源档落 sleep"，
   两者是**同一个动作**——因为面板在 `pstate` 分支里那句"库不存在就抛"排在启动引擎之前，缺库还留着 pstate 的后果是**引擎根本起不来**。
   包内那枚 580.173.02 副本保留但降级为 `NVAPI=package` 显式可选；它的失败模式是"宿主驱动一升级就哑"，
   这也是我们改默认值的原因（全链路证据与判据见 `docker/NVAPI-获取说明-v1.md`）。
6. **四笔要提前排的账**：① 换镜像后引擎**冷加载约 18 分半**（JIT/编译缓存空）、热缓存约 8 分钟——排切换窗口按这个量；
   ② 单条提示图数默认 16，超编码器缓存预算（恒 16384 tokens）会分波编码、时延明显涨（实测表在 `部署文档-v1.md`）；
   ③ 面板跑分的**聚合值不入库**，引用任何一格数字必须当场把面板那一行连同 `date -u` 时刻一起记下来，否则事后无法归属。
   别指望用 `nvidia-smi -lgc` 锁时钟省电：容器内实测 rc=4 无权限（Docker 默认能力集里没有 CAP_SYS_ADMIN），见口径红线那节。
   ④ 「模型测试」九类里 BugFind-15／CLI-40／HermesAgent-20 默认是灰的——**不是包里没有**，而是**部署侧没把三枚受管验证容器拉起来**
   （验证运行时与 9 个题包都在镜像里；起容器的官方脚本也已随包发在 `docker/benchlocal.sh`；面板容器里没有 docker CLI，
   它自己起不了，拉起要在宿主做）。要点亮就走 `部署文档-v1.md` 步骤 7 的四个动作（**不需要 `--network host`，
   也不需要把 docker.sock 挂进面板容器**）；不跑这三类就照默认六类 90 题用。

## 6. 许可

代码部分 Apache-2.0（见 `LICENSE`、`NOTICE`）。`docker/libnvidia-api.so.1` 是 NVIDIA 驱动侧运行库，
720104 字节、sha256 `4a199f9b…131d8c`、与驱动 580.173.02 配套，**v1.2 起只是离线备胎**（默认改用宿主驱动自带那份，
要显式用它才 `NVAPI=package`）；它仍受 NVIDIA 驱动许可约束、不由本包授权再分发，本包也不主张第三方可以再分发它。
不想用这个二进制：包 v1.2 起默认就不碰它（宿主探不到库时自动落 sleep），而且包内没有它也不再拦构建（只 WARN）。
取件与判据见 `docker/NVAPI-获取说明-v1.md`。
模型权重不在包内，第三方量化档有自己的许可与基座模型条款，用前请自行核对。
