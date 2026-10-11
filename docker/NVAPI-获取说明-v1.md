# NVAPI 库说明（`libnvidia-api.so.1` · 2026-10-11 · 0.1.7 包线）

## 三件事先说清

1. **0.1.7 这条线不把它烘进镜像**。0.1.6 包把它做成了一个镜像层（`Dockerfile.nvapi`），0.1.7 官方形态是
   **宿主文件只读 bind** 进容器（`run/start-here-v2.sh` 里那条 `$NVAPI:/usr/local/nvidia/lib64/libnvidia-api.so.1:ro`）。
2. **包 v1.2 起，默认用宿主驱动自带那一份**（`NVAPI=auto`）。包内那枚 `docker/libnvidia-api.so.1`
   降级成**显式 opt-in**（`NVAPI=package`），文件保留、一个字节没动。
3. 它只服务控制台的 **P-State 电源托管**。推理本体（权重加载、生成）与它无关，它也不改吞吐。
   **但"缺它"的代价不是"少个省电功能"**——见下面那条因果，写死在这里。

## 为什么默认要改成"探宿主"（都是实机证据，不是推测）

包内那枚是驱动 **580.173.02** 的私有副本（720,104 字节，sha256 `4a199f9b…131d8c`，与发布机当年自己钉的那份同一个文件）。
发布机的宿主驱动被 `unattended-upgrade` 从 580.173.02 升到 **580.178.04** 之后：

- 容器内 `nvidia-pstated` 每次在 **0.1 秒内**以 `NvAPI_Initialize(): NVAPI_ERROR` 退出；升级后 6/6 全灭、升级前 6/6 打出 banner。
- 表现是 **P-State 静默失效**：配置读着全对、面板日志无错、只有执行体死了。
- 而这枚副本**不归 dpkg 管**——apt 升级驱动时既不会更新它、也不会删它。它只有一种失败模式：**宿主驱动一漂移就哑**。

另一条硬事实：**Ubuntu 的 deb 驱动包根本不携带 NVAPI**，所以"从宿主驱动包里取"这条路在 Ubuntu server 上取不到件：

| 检查 | 结果 |
|---|---|
| `dpkg -L libnvidia-compute-580-server` | 20 条 `.so`，**无 api** |
| `dpkg -S libnvidia-api.so.1` | **无匹配** |
| `apt-cache policy libnvidia-api1 libnvidia-tesla-api1` | **无候选** |
| `nvidia-container-cli list --libraries` | 9 条全为 `.580.178.04`，**无 api** |
| 宿主 `libnvidia-ml/cuda/cfg/opencl` 四件内嵌 `NvAPI_` 符号计数 | **全 0**（包内那枚副本＝3） |
| 换非 server flavor 有没有 | **也没有**：三对同名包 Installed-Size 逐枚相等、deb 体积只差 228/94/18 字节，而那枚库有 704 KiB |

⇒ 上游作者那句"本来就用宿主驱动的文件"在 runfile／Arch／Debian 独立包上成立，在 Ubuntu server deb 上取不到件。
所以本包的做法是：**默认探宿主（探得到就跟宿主走），探不到就明确降级**，同时把那份 173.02 副本留作显式可选项。

## 默认值与四种取值

`run/start-here-v2.sh` 里 `NVAPI` 的取值（默认 `auto`）：

| 取值 | 含义 |
|---|---|
| `auto`（默认） | 探宿主自带那份，顺序照上游官方启动脚本那三条：`/usr/lib64` → `/usr/lib/x86_64-linux-gnu` → `/usr/lib`；探到就用它，探不到⇒ 当 `none` 处理并**同时**降级电源档 |
| `none` | 不带 P-State 库，电源档落 `sleep`（推理不受影响） |
| `package` | 显式用包内那枚 580.173.02 副本（sha 门照跑） |
| `/绝对路径` | 用你自己从驱动里取的那份 |

自己查宿主机上有没有、在哪：

```bash
nvidia-smi
ldconfig -p | grep -E 'libnvidia-(api|ml)\.so\.1'
# 链接器缓存里没有就逐条看：
ls -l /usr/lib64/libnvidia-api.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-api.so.1 /usr/lib/libnvidia-api.so.1
```

`ls` 里部分路径不存在是正常的，取实际存在**且可读**的那条（`NVAPI=auto` 也只认存在且可读的）。

## sha 门现在怎么判（三条，别再拿包内值去要求宿主那份）

`run/start-here-v2.sh` 对选定文件做三档判定：

1. 与包内钉死值 `4a199f9b…131d8c` **逐字节相同** ⇒ PASS；
2. 不同，但路径**就在上面那三条宿主库目录里** ⇒ PASS，并把实际 sha 与宿主驱动版本打印出来留痕
   （这一条是新加的：拿 173.02 的哈希去要求宿主 178.04 自带的库，等于每升一次驱动就自己把自己拦死）；
3. 既不是钉死值、也不在那三条目录里 ⇒ **BLOCK**：来路不明的同名文件不许 bind 进容器（供应链纪律，原样保留）。

`docker/build.sh` 里那道门也同步降了级：**包内副本不在位只 WARN、不再拦构建**（构建本来就不碰这个库）；
文件在时 sha 照核——它是对外分发的那枚二进制，被人改动过必须拦下来。

## 缺库而电源档还是 pstate ⇒ 引擎根本起不来（这条因果必须写死）

控制台代码里的次序是这样的：进 `pstate` 分支 → 校验 `/usr/local/nvidia/lib64/libnvidia-api.so.1` 存在，
不存在就**直接抛错** → 再往后才起电源监督进程、才 `launch("vllm", ["serve" …])`
（beta2 源码树里对应 console 目录 `standalone.mjs` 的第 170 行进分支、`173-176` 行校验并抛、`204` 行才起引擎）。

所以**"默认不挂库"与"电源档落 sleep"必须是同一个动作**，不是两件事。包 v1.2 里它是一体的：

| 层面 | 谁改 | 改什么 |
|---|---|---|
| 容器 env | `start-here-v2.sh` 自动（**只覆盖没有 env-file 那条路**） | `POWER_MODE=pstate` → `sleep`、`PSTATE_*` 不再下发；走 `--env-file` 时脚本改不到你那份 `console.env`，只补一条 `-e`，**实际生效值由终态自检断** |
| 包内模板档 | `start-here-v2.sh` 自动 | 摆进 `$DATA/tmp/profiles/` 的**那一份副本** `power.mode` 改 `sleep`（包内原件不动，那是对外基线） |
| 已经建好的档 | 你 | `bash run/configure-v1.sh` 的电源那一题（它**先查容器里库在不在**，不在就只让你选 sleep），或：`docker exec <容器> node /console-data/tmp/console-edit-profile-v3.cjs <profileId> setfield power.mode sleep` |

三条判据（按面板自己判的那条路径查，别看"我 `--volume` 过"）：

```bash
docker exec <容器> test -e /usr/local/nvidia/lib64/libnvidia-api.so.1 && echo 在 || echo 不在
docker exec <容器> python3 -c 'import ctypes; ctypes.CDLL("libnvidia-api.so.1"); ctypes.CDLL("libnvidia-ml.so.1"); print("NVAPI / NVML load OK")'
```

**注意第二条的证明范围**：ctypes 能加载只说明这枚 `.so` 与容器里的驱动侧库对得上，
**不证明 `NVAPI_Initialize` 成功**——580.173.02 的副本配 580.178.04 的宿主驱动时文件在位、加载不报错，
而 `nvidia-pstated` 每次 0.1 秒内退出。P-State 是否真生效要在引擎起来之后判（空闲进 P8、负载回 16），
命令见 `部署文档-v1.md` 步骤 5。

## 干脆不带这个文件

```bash
NVAPI=none bash run/start-here-v2.sh        # 起容器时不加那条 bind，电源档自动落 sleep
```

v1.2 起这一步**不需要再动 `docker/build.sh`**：包内副本不在位只 WARN。
`SKIP_NVAPI=1` 仍然认，但它现在只表示"连包内那枚物料的 sha 检查也一并跳过"，对启动形态没有直接影响。

代价：**没有这个库就没有 P-State 托管**，空闲时不会把卡拉到低功耗档。
包内两份 profile 模板默认都是 `power.mode=pstate`＋`gpus=0..7`（与官方模板一致），
`validateProfile` 里没有"关电源"这个选项，只有 `pstate` / `sleep` 两档。

## 要显式用包内那份

```bash
NVAPI=package bash run/start-here-v2.sh
```

前提是把 `docker/libnvidia-api.so.1` 留在包里原位。代价说清：**驱动一升级它就哑**（上面那一整节就是这个代价的实况），
所以它只该用于"宿主确实没有、又非要 P-State 不可"的场景，且换驱动后要主动复核 P-State 还活着。

## 从你自己的驱动取件

```bash
# 从任何已含该库的同族镜像提取（已实证可行的通道）
docker run --rm --entrypoint cat <含该库的镜像> /usr/local/nvidia/lib64/libnvidia-api.so.1 > libnvidia-api.so.1
sha256sum libnvidia-api.so.1     # 记下来，核对用的就是这个值
```

**这条通道不是 sha 门的旁路**。"路径在三条宿主库目录里就放行"那一档，认的是**宿主驱动/dpkg 自带的那一枚**；
自己从别处拷一枚进去、再靠路径躲开哈希核对，属于下面"红线"要拦的事。取件后请把它当"来源需要解释的文件"：
要么显式 `NVAPI=<你的路径>` 并留下 sha 记录，要么放回包内原位走 `NVAPI=package`（那条走钉死哈希）。

## 与上游口径的对拍（这一节是 v1.2 补的，用来界定"包说的"与"上游给的"是否一致）

- 上游文档（源码树 docker 目录里的 BUILD.md）第 110 行的表：`libnvidia-api.so.1` **由宿主只读挂载**，可用 `PSTATE_NVAPI_LIB` 指定；
  第 141 行：那三条路径**标准版自动检查、Ultra 必须显式指定**；第 194 行：**未挂载时在控制台选择 sleep 电源模式**。
  ⇒ 本包"0.1.7 形态＝宿主只读 bind"这句话**与上游一致**。
- 上游脚本：docker 目录里的 run.sh 第 107 行就是那三条路径（本包的探测顺序照它），第 112 行探不到是**直接 exit 2**；
  docker/helpers 里的 run-ultra.sh 第 17-19 行只认显式 `PSTATE_NVAPI_LIB`，不做自动探测。
  ⇒ **差异只有一处**：上游探不到就退出，本包探不到就降级成"不带库＋电源档落 sleep"。这是本包的选择，不是上游的行为。
- 上游文档第 152 行还有一句"库加载成功后，还需实际确认空闲进入 P8、负载恢复 16（驱动自动）"
  ⇒ 本包把它当**判据**引用（见上面那三条判据），不当结论。
- BenchLocal 那三类"受管验证"容器**不需要这个库**：上游起它们时显式给 `NVIDIA_VISIBLE_DEVICES=void`、不挂 GPU
  （见 `部署文档-v1.md` 步骤 7）⇒ 别把两件事混成一条依赖。

## 一条本机没证到的（别当结论用）

电源监督脚本的 **active 分支**在脚本层确实传 `-psl 16 -psh 16`（上游文档也称 16 是"驱动自动"），
但本机在 `nvidia-pstated` 二进制的字符串表里**只找到 `NvAPI_GPU_SetForcePstate`，没有任何 unforce/auto 调用**
⇒ "16＝把控制权交还驱动自动"这句按上游文档理解即可，**本机无法独立证实**，16 是否为自动哨兵值＝证据不足。

## 红线

- 不要从不明来源下载同名文件就 bind 进容器（第 3 条 sha 判法就是拦这个）。
  也**不要拿"路径放行"那一档当哈希核对的后门**：那一档只适用于宿主驱动自己装上去的那一枚。
- 别把它当"性能项"：它不改吞吐。它决定的是面板肯不肯按 pstate 起引擎、以及空闲功率档。
- 别把它当"驱动管理手段"：本包不 hold 驱动版本，也不建议为了迁就一枚私有副本去 `apt-mark hold`——
  那是拿系统安全更新换一个省电功能。
