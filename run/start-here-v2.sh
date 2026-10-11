#!/bin/bash
# start-here-v2.sh —— vLLM-SM75 0.1.7 自包含部署包的唯一启动入口
#
# v2 相对 v1 的三处变化（其余逐字保留）：
#   ① 进 P8 的默认值 86400 s → 120 s（官方文档自己写"部署后可按需要调小"，86400 是测速常驻档）；
#   ② 读 $DATA/config.env（由 run/configure-v1.sh 写）——按行按白名单取键，不 source；
#   ③ 拷进容器的改档工具升到 console-edit-profile-v2.cjs（多 addarg/delarg/setfield/setmodel/dump/list）。
#
# 包 v1.2 改了本脚本的两件事（文件名沿用 v2，语义沿革见 CHANGELOG v1.2；不另起 v3 是为了不留两个入口脚本）：
#   ① NVAPI 默认语义。
#   旧：默认 bind 包内 docker/libnvidia-api.so.1（钉死驱动 580.173.02 的私有副本），宿主没有它也照用；
#   新：默认 NVAPI=auto ⇒ 先探宿主自带的库（上游官方 docker/run.sh 那三条路径），
#       探不到就 NVAPI=none 并**同时**把电源档降成 sleep（BOOTSTRAP 里换 POWER_MODE、去掉 PSTATE_*；
#       走 --env-file 时这里改不到 console.env，所以第 2 步补一条 -e、终态由 5.8 读实际生效值断），
#       并把本次摆进容器的包内模板档副本的 power.mode 一起改成 sleep（缺库而档仍是 pstate ⇒
#       面板在启动引擎之前就抛，引擎根本起不来）。
#   包内那枚副本保留、降级为显式 opt-in（NVAPI=package），一个字节都没动。
#   ② 摆进容器的改档工具升到 console-edit-profile-v3.cjs（v2 的十二种模式逐字保留，只把 clone 的
#      新档 id 字符集收紧到上游删除路由那一套；v2 因此作废、仍留在 tools/panel/ 但不被引用）。
#
# 为什么要有这个脚本：控制台容器的启动参数一旦少一条（内存上限、shm、ulimit、某条缓存 bind），
# 表现不是"起不来"，而是"看起来一样但行为不同"——重付编译费，或者引擎被 OOM 杀。
# 所以全部显式化、每条进门禁；机器专属路径一律走环境变量，脚本里不写死。
#
# 用法：
#   DATA=/var/lib/sm75-console MODELS=/var/lib/sm75-models/Flash-Next-FP8PLE bash run/start-here-v2.sh
#   先看组装出来的命令、且确认零副作用：  ... bash run/start-here-v2.sh --dry-run
#
# 起完之后引擎不会自起（生产惯例，自起会在开机时抢卡）：去面板点「启动」，或
#   docker exec "$NAME" node /console-data/tmp/console-driver-v1.cjs start <profileId>
#   profileId 现取：docker exec "$NAME" node /console-data/tmp/console-driver-v1.cjs probe
#   （profiles.json 是 root 0600，只能走面板 API，别去读文件。这三个 .cjs 由本脚本从 tools/panel/ 摆到 $DATA/tmp/）
#
# 想让"重建容器后面板被 boot 抢走的默认档"自动落回来（机制见 部署文档-v1.md「步骤 4·删档」那一节）：
#   在 $DATA/config.env（或调用时的 env）里显式声明 SM75_WANTED_DEFAULT_PROFILE=<档 id>，
#   要连自动启动一起钉住就再加 SM75_WANTED_AUTOSTART=0|1。**两个键都不给 ⇒ 本脚本一行业动作都不做**，
#   只在终态自检里印一行 SKIP。默认行为＝沿用面板现值，这是有意的（默认值锁基线，适配只走覆盖口）。
#   工具失败按退出码 2/3/4/5 分别报，不做"跑完了就算成功"的处置。
set -u

P=$(cd "$(dirname "$0")/.." && pwd)          # 包根，tools/panel/ 相对它

# ---------- 参数来源优先级：调用方 env ＞ $CONFIG ＞ 包内默认 ----------
# 先记下"哪些键调用方已经在环境里给了"，否则 config.env 会把显式传的值悄悄盖掉。
# CONFIG 与 DATA 只认 env／默认值，不从 config.env 读——"配置文件在哪"不能靠它自己定义。
FROM_ENV=" "
for _k in NAME IMG MODELS CACHE NVAPI ENVFILE BIND_HOST PANEL_PORT API_PORT MEM_LIMIT SHM_SIZE RESTART PSTATE_IDLE_TIMEOUT SM75_WANTED_DEFAULT_PROFILE SM75_WANTED_AUTOSTART; do
  [ -n "${!_k:-}" ] && FROM_ENV="$FROM_ENV$_k "
done
CONFIG=${CONFIG:-${DATA:-/var/lib/sm75-console}/config.env}

# ---------- 可覆盖参数（默认值＝官方 0.1.7 文档口径；本机漂移见 基线与口径说明-v1.md） ----------
NAME=${NAME:-sm75-017-console}
IMG=${IMG:-vllm-sm75:v0.1.7-ultra-beta2-to20s-pcieipc}
DATA=${DATA:-/var/lib/sm75-console}                       # 面板数据目录（宿主侧，容器内是 /console-data）
MODELS=${MODELS:-/var/lib/sm75-models/Flash-Next-FP8PLE}  # 权重目录，只读挂成 /models
CACHE=${CACHE:-/var/lib/sm75-cache}                       # 四条 JIT/编译缓存的宿主根
# NVAPI：默认 auto＝**优先用宿主自带**那份（上游口径：P-State 的库要跟宿主驱动走，见 上游文档 docker/BUILD.md 第 110 行）。
# 为什么不默认用包内那份：docker/libnvidia-api.so.1 是驱动 580.173.02 的私有副本，它**不归 dpkg 管**，
#   宿主驱动被 unattended-upgrade 升到 580.178.04 之后，容器内 nvidia-pstated 每次在 0.1 秒内以
#   NVAPI_Initialize(): NVAPI_ERROR 退出——配置读着全对、日志无错、只有执行体死了，P-State 静默失效。
#   这种副本只有一种失败模式：宿主驱动一漂移就哑。所以它降级成显式 opt-in（NVAPI=package），文件保留不删。
# 探测的三条路径与顺序照上游官方 docker/run.sh 第 107 行；上游探不到是直接 exit 2，
#   本包改成"降级为不带库＋电源档同时落 sleep"（见下面 NVAPI 落地那一段与 BOOTSTRAP 那一段）。
# 取值：auto(默认)＝探宿主；none＝不带 P-State 库（档与容器都落 sleep）；
#       package＝显式用包内那枚 580.173.02 副本；/绝对路径＝用你自己从驱动里取的那份。
NVAPI=${NVAPI:-auto}
ENVFILE=${ENVFILE:-$DATA/console.env}                     # 有则用它；没有则用 BOOTSTRAP_ENV
BIND_HOST=${BIND_HOST:-127.0.0.1}                         # 端口默认只绑回环；要局域网访问显式改成 0.0.0.0 并自行加鉴权
PANEL_PORT=${PANEL_PORT:-1615}                            # 控制台 Web 对外端口
API_PORT=${API_PORT:-8000}                                # 引擎 OpenAI 兼容端口对外（基线 8000；本机 18001 属漂移项）
MEM_LIMIT=${MEM_LIMIT:-120259084288}                      # 112 GiB＝官方文档值；权重加载峰值实测 110.7/112，别调低
SHM_SIZE=${SHM_SIZE:-17179869184}                         # 16 GiB＝官方文档值
RESTART=${RESTART:-no}                                    # 生产惯例：用完即停、不自起
NVAPI_SHA=4a199f9b259a1098ab9c01d31c67f882a2531a0fbb9c3595ad3d016c7d131d8c

# $CONFIG 由 run/configure-v1.sh 写（面板端口、内存上限、P8 秒数这些）。这里**按行按白名单取键**，
# 不 source：config.env 是配置文件，不是可执行入口——source 一个被人改过的文件等于把 shell 交出去。
if [ -s "$CONFIG" ]; then
  n_cfg=0
  while IFS= read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    case "$line" in *=*) : ;; *) continue ;; esac
    k=${line%%=*}; v=${line#*=}
    case "$k" in
      NAME|IMG|MODELS|CACHE|NVAPI|ENVFILE|BIND_HOST|PANEL_PORT|API_PORT|MEM_LIMIT|SHM_SIZE|RESTART|PSTATE_IDLE_TIMEOUT|SM75_WANTED_DEFAULT_PROFILE|SM75_WANTED_AUTOSTART) : ;;
      *) continue ;;                                  # 白名单外的键直接丢
    esac
    # `+` 不是 `*`：BIND_HOST= （空值）必须被拦，否则拼出 `--publish ":1615:…"` ＝ docker 绑到全网卡，
    # 而端口暴露这件事要等到第 5 步的断言才红——容器已经带着敞开的口起来了。
    printf '%s' "$v" | grep -qE '^[A-Za-z0-9._:/,=-]+$' || { echo "BLOCK $CONFIG 里 $k 的值不合规（空值或含非法字符；只允许字母数字与 ._:/,=-）"; exit 4; }
    case "$FROM_ENV" in *" $k "*) continue ;; esac     # 调用方显式给的优先，不被文件盖掉
    printf -v "$k" '%s' "$v"
    n_cfg=$((n_cfg + 1))
  done < "$CONFIG"
  echo "  已读 $CONFIG：$n_cfg 项生效（调用方 env 覆盖的项保持原值）"
fi

# ---------- NVAPI=auto 在这里落地 ----------
# 为什么排在 config.env 之后：那一层也能给 NVAPI（configure 写的），"配置文件在哪"不能靠它自己定义，
# 但"用哪份库"是配置项，必须允许它被 config.env 覆盖之后再解析。
# 为什么排在 BOOTSTRAP 组装之前：下面那段"NVAPI=none ⇒ POWER_MODE 落 sleep"要靠这里的结论才会触发。
NVAPI_PKG=$P/docker/libnvidia-api.so.1
NVAPI_HOST_CANDIDATES=(/usr/lib64/libnvidia-api.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-api.so.1 /usr/lib/libnvidia-api.so.1)
NVAPI_FROM_HOST=no
if [ "$NVAPI" = "auto" ]; then
  nv_found=""
  for c in "${NVAPI_HOST_CANDIDATES[@]}"; do
    # -f 且 -r：只读 bind 要求宿主进程真能读它；不可读的条目当"没有"处理，别起一个必然挂的 bind
    if [ -f "$c" ] && [ -r "$c" ]; then nv_found="$c"; break; fi
  done
  if [ -n "$nv_found" ]; then NVAPI="$nv_found"; NVAPI_FROM_HOST=yes; nv_why="命中宿主自带那份，与宿主驱动同版本"
  else NVAPI=none; nv_why="三条标准路径都没有 ⇒ 本次不带 P-State 库，电源档同时落 sleep"; fi
  echo "  NVAPI=auto 解析：$NVAPI　（宿主探测：$nv_why）"
fi
if [ "$NVAPI" = "package" ]; then NVAPI="$NVAPI_PKG"; fi

# 最小 env 集（没有 console.env 时用）。值来自 2026-10-07 在役容器实读，不是抄文档：
#   镜像自己已带 SM75_EDITION/NCCL_P2P_LEVEL/VLLM_FIREFLY/NVIDIA_VISIBLE_DEVICES，这里只补面板行为项。
# SM75_CONSOLE_ROOT 必须是 /console-data（= bind 目标）。写成镜像默认的 /data 或别的路径，
# profiles.json 与登录 key 就会落在容器可写层里，重建容器即丢。
BOOTSTRAP_DEFAULT=(
  SM75_EDITION=ultra
  SM75_SINGLE_CONTAINER=1
  SM75_CONSOLE_ROOT=/console-data
  SM75_CONSOLE_HOST=0.0.0.0
  SM75_CONSOLE_PORT=1615
  POWER_MODE=pstate
  "PSTATE_GPUS=0,1,2,3,4,5,6,7"
  PSTATE_IDLE_TIMEOUT=120
  TZ=Asia/Shanghai
  NVIDIA_VISIBLE_DEVICES=all
)
# 覆盖口用分号而不是逗号：PSTATE_GPUS 这类值本来就含逗号，用逗号当分隔符会把值切碎，
# 碎出来的 "-e 1" 变成"透传宿主变量 1"＝静默丢值。切完还逐片验形状，不合规就 BLOCK。
# 约定：BOOTSTRAP_ENV="" 或没设 ⇒ 用包内默认集（这是安全侧，不是"什么都不发"）；
#       设了内容但不合规（例如只有分号）⇒ 门禁 BLOCK。静默零 env 起跑是最坏的形态。
BOOTSTRAP=()
if [ -n "${BOOTSTRAP_ENV:-}" ]; then
  IFS=';' read -ra BOOTSTRAP <<< "$BOOTSTRAP_ENV"
else
  BOOTSTRAP=("${BOOTSTRAP_DEFAULT[@]}")
fi
# NVAPI=none 却仍带 POWER_MODE=pstate ＋ PSTATE_* 是自相矛盾。**这一段的因果要说准**：
# 面板控制台的 pstate 分支里，那句"库不存在就 throw"排在**启动引擎之前**（standalone.mjs 进 pstate 分支
# → existsSync 校验不过直接抛 → 再往后才 spawn 监督进程、才 launch("vllm", ["serve" …)），
# 所以缺库而电源档还是 pstate 的表现**不是"少个省电功能"，而是引擎根本起不来**。
# 因此"不带库"与"落 sleep"必须是同一个动作：这里改 BOOTSTRAP 那一组，步骤 3 里同时改本次摆进容器的包内模板档副本。
# **但这里只覆盖了一半**：有 $ENVFILE 时第 2 步发的是 --env-file（整组 -e 都不发），console.env 里写着 pstate 就照旧是 pstate。
# 所以这条降级在 env-file 形态下**不由这里担保**——第 2 步会补一条 -e，终态由 5.8 读容器实际生效值来断（不赌 docker 的优先级）。
# 覆盖口 BOOTSTRAP_ENV 同样适用——自己传了 POWER_MODE=pstate 而没给库，这里一样换成 sleep。
# 已经建好的档则由 configure-v1.sh 的电源那一题（它会查容器里库在不在）或改档工具的 setfield 负责。
if [ "$NVAPI" = "none" ]; then
  nb=()
  for kv in "${BOOTSTRAP[@]}"; do
    case "$kv" in
      POWER_MODE=*) nb+=(POWER_MODE=sleep) ;;
      PSTATE_*) : ;;
      *) nb+=("$kv") ;;
    esac
  done
  BOOTSTRAP=("${nb[@]}")
fi
# 进 P8 的秒数有两处：容器 env 与档里的 power.idleSeconds。**两处都写同值，别赌谁赢。**
# 2026-10-09 真机实测（八卡当时都在 P8/645 MHz）：容器 Config.Env 是 86400，在役档里写 120，
# 而真正跑着的 pstate-supervisor 进程 environ 里是 **120** ⇒ 这台机上生效的是档值。
# 镜像里 runtime-entrypoint.py 用的是 env.setdefault，但 server.mjs 与 pstate-supervisor.sh 里都没有
# idleSeconds 字样，值到底从哪一路传下去**尚未定位** ⇒ 不写进包当结论，只按"两处同值"这条安全侧走。
# 权威判据是读 supervisor 进程的 environ（见 部署文档 步骤 5 的那条命令），不是读 Config.Env。
if [ -n "${PSTATE_IDLE_TIMEOUT:-}" ] && [ "${#BOOTSTRAP[@]}" != "0" ]; then
  ib=()
  for kv in "${BOOTSTRAP[@]}"; do
    case "$kv" in
      PSTATE_IDLE_TIMEOUT=*) ib+=(PSTATE_IDLE_TIMEOUT="$PSTATE_IDLE_TIMEOUT") ;;
      *) ib+=("$kv") ;;
    esac
  done
  if [ "${#ib[@]}" != "0" ]; then BOOTSTRAP=("${ib[@]}"); fi
  case " ${BOOTSTRAP[*]:-} " in *PSTATE_IDLE_TIMEOUT=*) : ;; *) BOOTSTRAP+=(PSTATE_IDLE_TIMEOUT="$PSTATE_IDLE_TIMEOUT") ;; esac
fi
# 注意：BOOTSTRAP_ENV=""（或只有分号）会切出空数组 ⇒ 一个 -e 都不发。这个后果留到门禁里判
# （见下面"最小 env 集"那一项），不在这里报错——那时 bad()/fail 还没定义，写了也不算数。

DRY=no
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=yes ;;
    *) echo "用法：bash run/start-here-v2.sh [--dry-run]"; exit 2 ;;
  esac
done

fail=no
good() { echo "  PASS $*"; }
bad() { echo "  BLOCK $*"; fail=yes; }
warn() { echo "  WARN  $*"; }
# --dry-run 什么都不写、什么都不起，所以只把"拦的是真起容器这一步"的两项（同名容器在册、卡被占用）
# 降级成 WARN，方便在生产机上半夜预演命令；其余门禁在 dry-run 里照样 BLOCK。
msg() { if [ "$DRY" = "yes" ]; then warn "$*"; else bad "$*"; fi; }

echo "=== 1) 门禁（任一 BLOCK ⇒ 立刻退出，本脚本一次写操作都不会做）==="
if docker image inspect "$IMG" >/dev/null 2>&1; then
  good "镜像 $(docker image inspect -f '{{.Id}}' "$IMG" | cut -c8-19) $IMG"
else bad "镜像不在位 $IMG（先跑 docker/build.sh；绝不隐式拉取）"; fi

# 同名容器：本包**不删、不停任何容器**（哪怕同名）。想复用名字就人工删掉再跑，或换个 NAME。
if docker inspect "$NAME" >/dev/null 2>&1; then
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = "true" ]; then
    msg "同名容器 $NAME 正在跑。处置：docker stop $NAME（确认它不是别人在用的），或换 NAME=新名字"
  else
    msg "同名容器 $NAME 已停但仍在册，docker run 必然 name conflict。处置：docker rm $NAME 后重跑，或换 NAME=新名字"
  fi
else good "容器名 $NAME 空闲"; fi

# 共存门：同一份 DATA 只能有一个控制台实例（profiles.json 会被整体重写覆盖）
sib=""
for cid in $(docker ps -q 2>/dev/null); do
  if docker inspect -f '{{range .HostConfig.Binds}}{{println .}}{{end}}' "$cid" 2>/dev/null | grep -q "^$DATA:"; then
    nm=$(docker inspect -f '{{.Name}}' "$cid" | sed 's|^/||')
    [ "$nm" = "$NAME" ] && continue
    sib="$sib $nm"
  fi
done
if [ -n "$sib" ]; then bad "已有容器占着 $DATA（同时跑会互相覆盖 profiles.json）:$sib"; else good "无其它控制台实例共享 $DATA"; fi

# 端口门：docker port 按**容器端口**列（形如 8000/tcp -> 0.0.0.0:18001），
# 所以必须匹配行尾的宿主端口；用 "^$宿主口/" 匹配永远匹配不到 —— 这是我们自己踩出的假放行。
port_holder() {
  for c in $(docker ps -q 2>/dev/null); do
    if docker port "$c" 2>/dev/null | grep -qE ":$1\$"; then
      docker inspect -f '{{.Name}}' "$c" | sed 's|^/||'; return 0
    fi
  done
}
for hp in "$PANEL_PORT" "$API_PORT"; do
  holder=$(port_holder "$hp")
  if [ -z "$holder" ] && command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -qE "[:.]${hp}[[:space:]]"; then
    holder="(宿主上有进程在听，非容器)"
  fi
  if [ -n "$holder" ]; then msg "宿主端口 $hp 被 $holder 占着（真起容器会撞端口；--dry-run 只看组装，故降为 WARN）"; else good "宿主端口 $hp 空闲"; fi
done

busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1>1000{n++} END{print n+0}')
n_gpu=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -dc '0-9')
if [ "${n_gpu:-0}" != "8" ]; then bad "nvidia-smi 读到 ${n_gpu:-0} 张卡，期望 8";
elif [ "${busy:-9}" != "0" ]; then msg "$busy 张卡仍占用>1GiB，先确认是谁在用（真起容器会和它抢卡）"; else good "八卡全空"; fi

# 权重目录：既接受"MODELS 就是模型目录"（我们生产即此形态），也接受"MODELS 是父目录"（官方模板 defaultModel 的写法）
if [ -f "$MODELS/config.json" ]; then
  good "权重目录在位（模型目录本身）$MODELS"
elif ls "$MODELS"/*/config.json >/dev/null 2>&1; then
  good "权重目录在位（父目录，内含 $(ls -d "$MODELS"/*/config.json 2>/dev/null | wc -l | tr -dc '0-9') 个模型）：$MODELS"
  warn "面板注册时要用子目录路径；官方模板 defaultModel=/models/<子目录名> 就是这种挂法"
else bad "在 $MODELS 及其一级子目录里都没找到 config.json —— 权重没就位，后面全部免谈"; fi

if [ -s "$ENVFILE" ]; then
  # 官方基线就是把引擎的 VLLM_API_KEY 放在 --env-file 里（umask 077 生成 secrets/api.env），
  # 所以这里**不拦密钥类变量**——拦它就等于否掉官方流程。要拦的是"权限敞开"与"值被打印出来"。
  perm=$(stat -c '%a' "$ENVFILE" 2>/dev/null || stat -f '%Lp' "$ENVFILE" 2>/dev/null || echo 未知)
  nsec=$(grep -cE '^[A-Za-z0-9_]*(API_KEY|TOKEN|SECRET|PASSWORD|PRIVATE_KEY|CREDENTIAL)=' "$ENVFILE")
  if [ "$perm" != "600" ] && [ "$perm" != "400" ]; then
    bad "env-file 权限是 $perm，期望 600（里面通常有 VLLM_API_KEY）：chmod 600 $ENVFILE"
  else
    good "env-file $(grep -c '^' "$ENVFILE") 条、权限 $perm（其中密钥类 $nsec 条，值不回显）"
    root=$(grep -E '^SM75_CONSOLE_ROOT=' "$ENVFILE" | tail -1 | cut -d= -f2-)
    if [ -z "$root" ]; then
      bad "env-file 里没有 SM75_CONSOLE_ROOT ⇒ 面板会用镜像内置的 /data，而本脚本把 $DATA 绑到 /console-data ⇒ 档与登录 key 落在没被 bind 的路径上。补一行 SM75_CONSOLE_ROOT=/console-data"
    elif [ "$root" != "/console-data" ]; then
      bad "env-file 里 SM75_CONSOLE_ROOT=$root，与本脚本的 bind 目标 /console-data 不一致 ⇒ 面板数据不落宿主。要么把它改成 /console-data，要么按官方布局把 $DATA 绑到 /data 并保留这个值（两处必须一致）"
    else good "env-file 里 SM75_CONSOLE_ROOT=/console-data（与 bind 目标一致）"; fi
  fi
else
  if [ "${#BOOTSTRAP[@]}" = "0" ]; then
    # 手滑写成 BOOTSTRAP_ENV="" 或 ";" 时会切出空数组：一个 -e 都不发，而其余门禁照样全绿 ⇒
    # 面板把 profiles.json 写进镜像默认的 /data 而不是 bind 里的 /console-data（＝B2 那个"重建即丢"）。
    bad "BOOTSTRAP_ENV 切出来是 0 项（分隔符是分号不是逗号）。处置：给它正常内容，或 unset 它改用包内默认集（${#BOOTSTRAP_DEFAULT[@]} 条）"
  else
    badenv=""
    secretenv=""
    for kv in "${BOOTSTRAP[@]}"; do
      printf '%s' "$kv" | grep -qE '^[A-Za-z_][A-Za-z0-9_]*=[^;]+$' || badenv="$badenv [$(printf '%s' "$kv" | cut -c1-24)…]"
      # 名字判据与 env-file 那一条同一套，只报变量名、绝不回显值
      printf '%s' "$kv" | grep -qE '^[A-Za-z0-9_]*(API_KEY|TOKEN|SECRET|PASSWORD|PRIVATE_KEY|CREDENTIAL)='         && secretenv="$secretenv [$(printf '%s' "$kv" | cut -d= -f1)]"
    done
    if [ -n "$badenv" ]; then
      bad "BOOTSTRAP_ENV 里有不合规片段（每项须为 KEY=VALUE，分隔符是分号不是逗号；值含逗号是允许的）:$badenv"
    elif [ -n "$secretenv" ]; then
      bad "BOOTSTRAP_ENV 里有密钥类变量:$secretenv —— 走 -e 会把值写进本脚本 stdout 与宿主 docker run 的 argv（谁 ps 都看得到）。处置：把它们放进 $ENVFILE（chmod 600，本脚本走 --env-file），或从覆盖口里去掉"
    else
      good "无 env-file，本次用最小 env 集 ${#BOOTSTRAP[@]} 条：$(printf '%s ' "${BOOTSTRAP[@]}" | cut -d= -f1 | tr -s ' ' ' ' | cut -c1-160)…"
      warn "跑起来后建议用 tools/gen-console-env-v1.sh 固化成 $ENVFILE（走 --env-file 值不进 argv），别再依赖默认集"
    fi
  fi
fi

# NVAPI 三道判法。**关键**：不能拿包内那枚的 sha 去要求宿主自带那份——那样换个驱动小版本就必然 BLOCK，
# 而"库跟宿主驱动走"正是本次要恢复的默认行为（那条 sha 门的原意是防"来路不明的同名文件"，不是防驱动升级）。
#   ① 与包内钉死值逐字节相同 ⇒ PASS（包内副本 NVAPI=package，或别人拷的同哈希文件）
#   ② 不同、但路径就在上游官方那三条宿主库目录里 ⇒ 放行并记录实际 sha：它是宿主驱动自带那份，
#      正确性来源是宿主驱动/dpkg，不在本包 sha 表的管辖范围里
#   ③ 既不是钉死值、也不在那三条目录里 ⇒ BLOCK：来源不明，供应链纪律不允许 bind 进容器
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -dc '0-9.')
if [ "$NVAPI" = "none" ]; then
  warn "NVAPI=none：本次不挂 P-State 库。电源档的降级分两条路——没有 env-file 时上面那段已把 BOOTSTRAP 里的 POWER_MODE 换成 sleep、并把 PSTATE_ 开头的两条一并撤掉（电源档已是 sleep，再带着 pstate 专用的变量就是自相矛盾；不是丢件）；"
  warn "  有 env-file 时本脚本只补一条 -e，**实际生效值由终态自检 5.8 断**（console.env 是你的文件，脚本改不得）"
  warn "  但**已经建好的档**里的 power.mode 是另一处：它仍是 pstate 的话，面板会在启动引擎**之前**就抛错——"
  warn "  表现是引擎根本起不来，而不是少个省电功能。改法两条：bash run/configure-v1.sh 的电源那一题（它会先查库在不在），"
  warn "  或 docker exec $NAME node /console-data/tmp/console-edit-profile-v3.cjs <profileId> setfield power.mode sleep"
elif [ ! -f "$NVAPI" ]; then
  bad "缺 NVAPI 文件 $NVAPI（这是显式给的路径，不在位就停下）。处置：改成宿主实际存在的那份、或 NVAPI=none 走 sleep，取件办法见 docker/NVAPI-获取说明-v1.md"
else
  got=$(sha256sum "$NVAPI" | cut -d' ' -f1)
  on_host_path=no
  for c in "${NVAPI_HOST_CANDIDATES[@]}"; do
    [ "$NVAPI" = "$c" ] && on_host_path=yes
  done
  if [ "$got" = "$NVAPI_SHA" ]; then
    good "NVAPI 与包内钉死那份逐字节相同（路径 $NVAPI，宿主驱动 ${drv:-读不到}）"
  elif [ "$on_host_path" = "yes" ]; then
    good "NVAPI 用宿主驱动自带那份（$NVAPI，宿主驱动 ${drv:-读不到}，sha256 $got）：与包内钉死值不同属正常，它跟宿主驱动走"
    [ "$NVAPI_FROM_HOST" = "yes" ] || warn "  这一份是显式 NVAPI= 指过来的；默认 NVAPI=auto 探到的就是它，这个变量可以不给"
  else
    bad "NVAPI 既不是包内钉死那份、也不在上游官方那三条宿主库目录里（$NVAPI）⇒ 来源不明，不 bind 进容器。处置：NVAPI=auto 探宿主、NVAPI=package 用包内那份，或按 docker/NVAPI-获取说明-v1.md 从自己驱动取件"
  fi
fi

# 余量分两块看：镜像层落在根盘，profiles/缓存/env-file 落在 $DATA 与 $CACHE 所在盘——
# 只查根盘会漏（DATA 指到 /data 这种另一块盘的路径时，根盘余量说明不了它能不能写）。
nearest_exists() {
  d="$1"
  while [ ! -d "$d" ] && [ "$d" != "/" ]; do d=$(dirname "$d"); done
  printf '%s' "$d"
}
avail=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
if [ "${avail:-0}" -ge 40 ]; then good "根盘余量 $avail GiB（镜像层用）"; else bad "根盘余量 $avail GiB < 40"; fi
for where in "$DATA" "$CACHE"; do
  tgt=$(nearest_exists "$where")
  a2=$(df -BG --output=avail "$tgt" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ -z "$a2" ]; then warn "$where 所在盘读不到余量（df 失败），跳过这一项"
  elif [ "$tgt" = "/" ]; then good "$where 就在根盘（$a2 GiB）"
  elif [ "$a2" -ge 20 ]; then good "$where 所在盘（$tgt）余量 $a2 GiB"
  else bad "$where 所在盘（$tgt）余量只有 $a2 GiB < 20（面板数据与四条缓存都写这里）"; fi
done

# P8 秒数有三处来源：env-file、config.env（本脚本读）、档里的 power.idleSeconds。
# 有 env-file 时本脚本只发 --env-file、不再逐条 -e ⇒ env-file 那一行赢，这里只报不一致。
if [ -s "$ENVFILE" ]; then
  piv=$(grep -E '^PSTATE_IDLE_TIMEOUT=' "$ENVFILE" 2>/dev/null | tail -1 | cut -d= -f2-)
  if [ -n "${piv:-}" ] && [ "$piv" != "${PSTATE_IDLE_TIMEOUT:-120}" ]; then
    warn "env-file 里 PSTATE_IDLE_TIMEOUT=$piv，本次默认 ${PSTATE_IDLE_TIMEOUT:-120} —— 有 env-file 时以 env-file 为准，要统一就改它或删那行"
  fi
fi

# 包内面板 CLI 工具（B10：不发这两个文件，别人照包跑就点不亮那个后端）
for f in console-driver-v1.cjs console-edit-profile-v3.cjs console-set-default-v1.cjs; do
  [ -s "$P/tools/panel/$f" ] || bad "包里缺 tools/panel/$f"
done
# 包内两靶档：configure 的 clone（建第二靶档）从容器里读它们，缺了就是"文档有、路径无"
ls "$P"/run/profiles/*.json >/dev/null 2>&1 || bad "包里 run/profiles/ 没有一份 JSON 档"

if [ "$fail" = "yes" ]; then
  echo "GATE_FAIL 上面 BLOCK 项已清零之前，本脚本没有做过任何写操作（没建目录、没 chmod、没拷文件、没起容器）"
  exit 4
fi

echo "=== 2) 组装 docker run（纯计算，不写任何东西）==="
ENVARGS=()
if [ -s "$ENVFILE" ]; then
  ENVARGS=(--env-file "$ENVFILE")
  # 有 env-file 时 BOOTSTRAP 整组不发 ⇒ config.env / 调用方给的 PSTATE_IDLE_TIMEOUT 会静默丢掉。
  # 这里单独把它补成 -e。docker 的 `-e` 与 `--env-file` 谁赢，本机没守护进程可实测、也不去赌——
  # 第 5.6 步直接读容器实际生效值断言，不符就红（把"未文档化的优先级"变成"可验证的终态"）。
  if [ -n "${PSTATE_IDLE_TIMEOUT:-}" ]; then
    ENVARGS+=(-e "PSTATE_IDLE_TIMEOUT=$PSTATE_IDLE_TIMEOUT")
  fi
  # NVAPI=none 的电源档降级在 env-file 形态下**不是这里担保的**：上面那段只改得到 BOOTSTRAP。
  # 所以补一条 -e POWER_MODE=sleep，而"它有没有赢过 env-file 里那行 pstate"**不赌**——5.8 读容器实际生效值断，
  # 还是 pstate 就红，并把该改哪一行告诉你（与 PSTATE_IDLE_TIMEOUT 同一套处置思路）。
  if [ "$NVAPI" = "none" ]; then
    ENVARGS+=(-e "POWER_MODE=sleep")
  fi
else
  for kv in "${BOOTSTRAP[@]}"; do ENVARGS+=(-e "$kv"); done
fi
PUBS=(--publish "$BIND_HOST:$PANEL_PORT:1615/tcp" --publish "$BIND_HOST:$API_PORT:8000/tcp")
VOL=(--volume "$MODELS:/models:ro"
     --volume "$DATA:/console-data"
     --volume "$CACHE/root-cache:/root/.cache"
     --volume "$CACHE/triton:/root/.triton"
     --volume "$CACHE/nv:/root/.nv"
     --volume "$CACHE/tilelang:/root/.tilelang")
if [ "$NVAPI" != "none" ]; then
  VOL+=(--volume "$NVAPI:/usr/local/nvidia/lib64/libnvidia-api.so.1:ro")
fi
# 镜像的 ENTRYPOINT 已经是 "python3 /opt/vllm-sm75/runtime-entrypoint.py"、CMD 为空，
# 所以这里**只传镜像名**：再把那两句当命令传一遍，它们就会被当成 vllm 的参数吃掉。
CMD=(docker run -d --name "$NAME" --gpus all
     --memory "$MEM_LIMIT" --memory-swap "$MEM_LIMIT" --shm-size "$SHM_SIZE"
     --restart "$RESTART" --network bridge
     --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576
     "${PUBS[@]}" "${ENVARGS[@]}" "${VOL[@]}"
     "$IMG")
printf '  %q ' "${CMD[@]}"; echo

if [ "$DRY" = "yes" ]; then
  echo "DRY_RUN_OK 上面就是将要执行的命令。将要写的东西（本次一律没做）："
  echo "    mkdir -p $CACHE/{root-cache,triton,nv,tilelang} 与 $DATA/tmp/profiles；chmod 700 $DATA"
  echo "    cp tools/panel/*.cjs -> $DATA/tmp/；cp run/profiles/*.json -> $DATA/tmp/profiles/（NVAPI=none 时这些副本的电源档会同时被改成 sleep，包内原件不动）"
  echo "  本次没有建目录、没有 chmod、没有拷文件、没有起容器。"
  exit 0
fi

echo "=== 3) 目录与面板 CLI 工具就绪（门禁通过且非 dry-run 才动手写）==="
# 四条缓存 bind 必须全在位，少一条重建容器就重付编译费（.tilelang 最容易漏）
for d in root-cache triton nv tilelang; do mkdir -p "$CACHE/$d" || { echo "WRITE_FAIL 建不出缓存目录 $CACHE/$d"; exit 3; }; done
mkdir -p "$DATA/tmp" || { echo "WRITE_FAIL 建不出 $DATA/tmp"; exit 3; }
chmod 700 "$DATA" 2>/dev/null
mkdir -p "$DATA/tmp/profiles" || { echo "WRITE_FAIL 建不出 $DATA/tmp/profiles"; exit 3; }
for f in console-driver-v1.cjs console-edit-profile-v3.cjs console-set-default-v1.cjs; do
  cp "$P/tools/panel/$f" "$DATA/tmp/$f" || { echo "WRITE_FAIL 拷不进 $DATA/tmp/$f"; exit 3; }
done
# 包内两靶档也摆进容器可读的 bind：configure 的 clone 模式（建第二靶档）唯一的取件通道就是这里，
# 不拷的话那两份 JSON 是惰性文件——文档承诺了功能，脚本却没给路径。
n_pf=0; n_sw=0
for j in "$P"/run/profiles/*.json; do
  [ -f "$j" ] || continue
  dst="$DATA/tmp/profiles/$(basename "$j")"
  cp "$j" "$dst" || { echo "WRITE_FAIL 拷不进 $(basename "$j")"; exit 3; }
  # 本次不带 P-State 库 ⇒ **摆进容器的这一份副本**同时落 sleep（包内原件一个字节不动，那是对外基线）。
  # 不留"记得手工改"的坑：档一旦按 pstate 建出来而容器里没那枚库，面板在启动引擎之前就抛错。
  # 判据断终态而不信 sed：既断"sleep 真在"，也断"pstate 不残留"（只断前者会放过半改状态）。
  if [ "$NVAPI" = "none" ]; then
    sed -i 's/"mode": "pstate"/"mode": "sleep"/' "$dst" || { echo "WRITE_FAIL 改不动 $dst 的电源档"; exit 3; }
    grep -q '"mode": "sleep"' "$dst" || { echo "ASSERT_FAIL $dst 的电源档没换成 sleep"; exit 7; }
    # 残留判据只盯**这一处字段本身**（`"mode": "pstate"`），不做整文件级 grep——
    # 否则日后档里任何带 pstate 字样的新字段都会让这条永远红（假红比漏红更难查）。
    if grep -q '"mode": "pstate"' "$dst"; then echo "ASSERT_FAIL $dst 的电源档还是 pstate"; exit 7; fi
    n_sw=$((n_sw + 1))
  fi
  n_pf=$((n_pf + 1))
done
[ "$n_pf" != "0" ] || { echo "WRITE_FAIL 包内 run/profiles/ 一份 JSON 都没有"; exit 3; }
sw_note=""
[ "$n_sw" != "0" ] && sw_note="，其中 $n_sw 份的电源档随 NVAPI=none 落成 sleep"
echo "  已就绪：$CACHE/{root-cache,triton,nv,tilelang}、$DATA(700)、$DATA/tmp/{console-driver-v1.cjs,console-edit-profile-v3.cjs,console-set-default-v1.cjs}、$DATA/tmp/profiles/($n_pf 份档$sw_note)"

echo "=== 4) 起容器并等面板有响应（最多 3 分钟）==="
"${CMD[@]}" || { echo "RUN_FAIL docker run 非零退出"; exit 5; }
ok=no
for i in $(seq 1 36); do
  code=$(docker exec "$NAME" curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:1615/" 2>/dev/null)
  if [ "$code" = "200" ] || [ "$code" = "302" ]; then ok=yes; echo "  PANEL_UP try=$i code=$code"; break; fi
  sleep 5
done
[ "$ok" = "yes" ] || { echo "BLOCK 面板 3 分钟没响应，容器日志尾 30 行："; docker logs --tail 30 "$NAME" 2>&1; exit 6; }

echo "=== 5) 终态自检（断言产物，不是打印看看）==="
check_fail=no
cf() { echo "  ASSERT_FAIL $*"; check_fail=yes; }

# 5.1 端口映射真在位，且绑的是预期地址（BIND_HOST=127.0.0.1 时外部机器连不上，这是有意的）
maps=$(docker port "$NAME" 2>/dev/null)
for cp in 1615 8000; do
  if printf '%s' "$maps" | grep -qE "^$cp/tcp -> $BIND_HOST:"; then
    echo "  PASS 映射 $cp/tcp -> $BIND_HOST:$(printf '%s' "$maps" | awk -v c="$cp" '$1 ~ "^"c"/"{split($3,a,":");print a[2]}')"
  else cf "缺少 $cp/tcp -> $BIND_HOST: 的映射（实际：$(printf '%s' "$maps" | tr '\n' ' '))"; fi
done

# 5.2 PCIe-IPC 层：镜像带 sm75.t11a.component 标签才断言 hasattr=True（不带该层的镜像 False 是正常的，不能当失败）
if [ "$(docker image inspect -f '{{index .Config.Labels "sm75.t11a.component"}}' "$IMG" 2>/dev/null)" = "flashinfer-pcie-ipc-backport" ]; then
  hv=$(docker exec "$NAME" python3 -c \
        'import flashinfer.comm as c;print("True" if hasattr(c,"PcieIpcAllReduceWorkspace") else "False")' 2>/dev/null)
  if [ "$hv" = "True" ]; then echo "  PASS hasattr(flashinfer.comm,'PcieIpcAllReduceWorkspace') = True（vLLM 的判据）"
  else cf "镜像自称带 pcie-ipc 层，但 hasattr=$hv"; fi
else
  warn "该镜像没有 sm75.t11a.component 标签（未回填 PCIe-IPC）⇒ 跳过 hasattr 断言，后端会是 FIREFLY_AR"
fi

# 5.3 20s 层：同理按标签决定断言与否
if [ "$(docker image inspect -f '{{index .Config.Labels "sm75.localpatch"}}' "$IMG" 2>/dev/null)" = "gpu-probe-timeout-3s-to-20s" ]; then
  n=$(docker exec "$NAME" grep -cE 'timeout: 20000([^0-9]|$)' /opt/sm75-workbench/console/server.mjs 2>/dev/null)
  if [ "${n:-0}" = "1" ]; then echo "  PASS server.mjs 内 timeout: 20000 恰好 1 处"
  else cf "server.mjs 的 timeout: 20000 命中 ${n:-0} 处（期望 1）"; fi
else
  warn "该镜像没有 sm75.localpatch 标签（未打 20s 层）⇒ 跳过断言；本机 hardware.py 要 6.0–6.9 s，面板可能拒启"
fi

# 5.4 容器内 /console-data 确实指向那条 bind（看挂载表，不看"我 --volume 过"）
if docker inspect -f '{{range .Mounts}}{{.Destination}}{{println .}}{{end}}' "$NAME" 2>/dev/null | grep -qx "/console-data"; then
  echo "  PASS /console-data 在挂载表里"
else cf "/console-data 不在挂载表里（面板数据会落在容器可写层）"; fi

# 5.5 面板 CLI 工具与包内档模板真的在容器里看得见
for f in console-driver-v1.cjs console-edit-profile-v3.cjs console-set-default-v1.cjs; do
  if docker exec "$NAME" test -s "/console-data/tmp/$f"; then echo "  PASS 容器内可见 /console-data/tmp/$f"
  else cf "/console-data/tmp/$f 在容器里看不到（bind 映射没生效？）"; fi
done
npf=$(docker exec "$NAME" sh -c 'ls /console-data/tmp/profiles/*.json 2>/dev/null | wc -l' | tr -dc '0-9')
if [ "${npf:-0}" != "0" ]; then echo "  PASS 容器内可见 /console-data/tmp/profiles/（$npf 份，configure 的 clone 靠它）"
else cf "/console-data/tmp/profiles/ 里没有 JSON ⇒ 第二靶子建不出来"; fi

# 5.6 P8 秒数：读容器的 Config.Env。注意这条**只证明"docker 被告知的是哪个值"**，
#     不证明"引擎 supervisor 实际用的是哪个"——2026-10-09 真机实测过：Config.Env 是 86400 而
#     supervisor 进程 environ 是 120（档值生效）。要判"实际生效"就读 supervisor 的 environ，命令见部署文档步骤 5。
if [ -n "${PSTATE_IDLE_TIMEOUT:-}" ]; then
  got_p=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$NAME" 2>/dev/null \
          | grep -E '^PSTATE_IDLE_TIMEOUT=' | tail -1 | cut -d= -f2-)
  if [ "$got_p" = "$PSTATE_IDLE_TIMEOUT" ]; then echo "  PASS 容器 env 里 PSTATE_IDLE_TIMEOUT=$got_p（只证 docker 被告知值；实际生效看 supervisor 的 environ）"
  else cf "容器 env 里 PSTATE_IDLE_TIMEOUT=${got_p:-（没有这一条）}，与本次要求的 $PSTATE_IDLE_TIMEOUT 不符（多半是 $ENVFILE 里那行赢了过去——把它改成同值或删掉，再重建容器）"; fi
fi

# 5.7 P-State 库：断的是**面板自己判的那条路径**，不是"我 --volume 过"。
#     缺库而档还是 pstate ⇒ 面板在启动引擎之前就抛（standalone.mjs 进 pstate 分支后 existsSync 不过就 throw，
#     那句 throw 排在 launch("vllm", ["serve" …) 之前）⇒ 这一条管的是"引擎能不能起来"，不是省不省电。
PSTATE_LIB=/usr/local/nvidia/lib64/libnvidia-api.so.1
if [ "$NVAPI" = "none" ]; then
  if docker exec "$NAME" test -e "$PSTATE_LIB" 2>/dev/null; then
    warn "本次没挂库，但容器里 $PSTATE_LIB 存在（不是本脚本挂的——来源是镜像自带或 toolkit 注入，本包不判它跟宿主驱动配不配）"
    warn "  这种状态下 pstate 分支的存在性校验会过，但本次摆出的模板档已落 sleep；要按 pstate 用就先确认那枚库的来源与宿主驱动版本相符"
  else
    echo "  PASS 容器内确无 $PSTATE_LIB（与 NVAPI=none 一致，本次模板档已同步落成 sleep）"
  fi
else
  if docker exec "$NAME" test -e "$PSTATE_LIB" 2>/dev/null; then
    echo "  PASS 容器内可见 $PSTATE_LIB（面板 pstate 分支那条存在性校验能过）"
    # ctypes 能加载只证"这枚 .so 与容器里的驱动侧库对得上"，**不证 NVAPI_Initialize 成功**：
    # 580.173.02 的副本配 580.178.04 的宿主驱动时文件在位、加载不报错，而 nvidia-pstated 每次 0.1 秒内退出。
    # P-State 是否真生效要在引擎起来之后判，命令见 部署文档-v1.md 步骤 5。
    if docker exec "$NAME" python3 -c 'import ctypes; ctypes.CDLL("libnvidia-api.so.1")' >/dev/null 2>&1; then
      echo "  PASS 容器内 ctypes 可加载 libnvidia-api.so.1（上游 docker/BUILD.md 给的同一条检查；它不证明 NVAPI_Initialize）"
    else
      cf "$PSTATE_LIB 在位但容器内加载不了（版本/ABI 错配这一类）⇒ P-State 会静默失效，见 部署文档-v1.md 步骤 5 的判据"
    fi
  else cf "$NVAPI 说要挂进容器，但容器内看不到 $PSTATE_LIB（bind 映射没生效？）"; fi
fi

# 5.8 电源档终态（只在没挂库时断）：容器**实际生效**的 POWER_MODE 必须是 sleep 或干脆没有这一条。
#     为什么单独断：console.env 是用户的文件、本脚本改不得，而 docker 的 `-e` 与 `--env-file` 谁赢没有权威说法
#     ⇒ 不赌优先级，只断结果。这条不通就是本次改动最想防的那个形态（缺库＋pstate＝引擎起不来，却一路看着正常）。
if [ "$NVAPI" = "none" ]; then
  pm=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$NAME" 2>/dev/null \
        | grep -E '^POWER_MODE=' | tail -1 | cut -d= -f2-)
  if [ -z "$pm" ] || [ "$pm" = "sleep" ]; then
    echo "  PASS 容器实际生效 POWER_MODE=${pm:-（没设，面板按自己的默认走）}（与不带库一致）"
  else
    cf "容器实际生效 POWER_MODE=$pm 而本次没挂 P-State 库 ⇒ 面板会在启动引擎之前就抛，引擎起不来。处置：把 $ENVFILE 里 POWER_MODE 那行改成 sleep 或直接删掉（缺库时本包不该再下发 pstate），然后重建容器；要按 pstate 用就照 docker/NVAPI-获取说明-v1.md 把库挂上"
  fi
fi

# 5.9 期望默认档：**显式声明了才动**，没声明就一行业动作都不做（只印 SKIP）。
#     机制与判据的正文在 部署文档-v1.md「步骤 4·删档」那一节，这里只写这条步序自己的三件事：
#     ① 顺序门——本步排在第 4 步"面板答话"之后。server.mjs 的启动导入跑在 listen 之前
#        （本地 beta2 源码树 ultra/source/console/server.mjs 的 `if (boot ||` 那一段），
#        所以只有等它答话再 POST，才不会刚改回来就被那一路压回去。第 4 步用的是有 deadline 的轮询，不是固定 sleep。
#     ② 只发需要的字段——store.saveSettings 是 `{...旧, ...新}` 的合并语义（store.mjs:20），
#        所以 defaultProfile（以及显式给了才带的 autoStart）之外一个键都不发，不去覆盖别人的 modelRoot/cacheRoot/source。
#     ③ 断到回读——工具自己 GET 一次设置并与期望比对，退出码 5＝"POST 过去了但值不是这个"；
#        本脚本再按码逐条处置。只看"命令跑完了"或只看 HTTP 200 都不算数（空壳路由那次的教训）。
if [ -n "${SM75_WANTED_AUTOSTART:-}" ] && [ -z "${SM75_WANTED_DEFAULT_PROFILE:-}" ]; then
  cf "声明了 SM75_WANTED_AUTOSTART=$SM75_WANTED_AUTOSTART 却没声明 SM75_WANTED_DEFAULT_PROFILE ⇒ 只推 autoStart 会被上游那道「自动启动需要指定默认配置」拒，两个键要一起给（本次没动设置）"
fi
if [ -n "${SM75_WANTED_DEFAULT_PROFILE:-}" ]; then
  sd_shape_ok=yes
  printf '%s' "$SM75_WANTED_DEFAULT_PROFILE" | grep -qE '^[a-z0-9][a-z0-9-]{0,47}$' \
    || { cf "SM75_WANTED_DEFAULT_PROFILE 的形状不合规（只能小写字母／数字／连字符、48 字符内——这是上游删除路由与改档工具共用的字符集）：$SM75_WANTED_DEFAULT_PROFILE"; sd_shape_ok=no; }
  if [ -n "${SM75_WANTED_AUTOSTART:-}" ] && [ "$SM75_WANTED_AUTOSTART" != "0" ] && [ "$SM75_WANTED_AUTOSTART" != "1" ]; then
    cf "SM75_WANTED_AUTOSTART 只能是 0 或 1（填了 $SM75_WANTED_AUTOSTART）"; sd_shape_ok=no
  fi
  if [ "$sd_shape_ok" = "yes" ]; then
    sd_args=("$SM75_WANTED_DEFAULT_PROFILE")
    [ -n "${SM75_WANTED_AUTOSTART:-}" ] && sd_args+=("$SM75_WANTED_AUTOSTART")
    sd_out=$(docker exec "$NAME" node /console-data/tmp/console-set-default-v1.cjs "${sd_args[@]}" 2>&1); sd_rc=$?
    case "$sd_rc" in
      0) echo "  PASS $sd_out" ;;
      2) cf "期望默认档这步没做成：参数被工具拒（形状不合规）—— $sd_out" ;;
      3) cf "期望默认档这步没做成：那枚档不在面板库里（改名了？还是这台机上没建它）—— $sd_out。列档：docker exec $NAME node /console-data/tmp/console-edit-profile-v3.cjs - list" ;;
      4) cf "期望默认档这步没做成：面板设置路由没答话或被拒 —— $sd_out" ;;
      5) cf "期望默认档这步没做成：POST 之后回读不等于期望 —— $sd_out" ;;
      *) cf "期望默认档这步没做成：工具退出码 $sd_rc（输出：$sd_out）" ;;
    esac
  else
    echo "  形状没过关 ⇒ 本次不向面板发任何设置请求（上面那条 ASSERT_FAIL 要先解决）"
  fi
else
  echo "  SKIP 未声明 SM75_WANTED_DEFAULT_PROFILE ⇒ 本次不改面板设置，默认档维持面板现值（一行业动作都没做）"
fi

if [ "$check_fail" = "yes" ]; then
  echo "DONE_WITH_FAILS 容器起了但终态断言没过 —— 别按成功用；上面 ASSERT_FAIL 每条都要解决或显式撤回该层"
  exit 7
fi
echo "RUN_DONE 全部终态断言通过"
echo "  浏览器：http://$BIND_HOST:$PANEL_PORT　　API：http://$BIND_HOST:$API_PORT/v1"
[ "$BIND_HOST" = "127.0.0.1" ] && echo "  默认只绑回环：要局域网访问请显式 BIND_HOST=0.0.0.0 重跑，并自行加 --api-key 与网络边界控制"
echo "  起引擎：面板点「启动」，或 docker exec $NAME node /console-data/tmp/console-driver-v1.cjs probe 取 profileId 后 start"
echo "  引擎就绪判据（日志）：dispatch 列表含 FLASHINFER_PCIE_IPC、Initialized FlashInfer PCIe IPC all-reduce、"
echo "    GPU KV cache size: 339,110 tokens、FP8 layout verified 96 条、backend=p2p 8 条、无 Traceback。"
echo "  撤回填层：SKIP_PCIEIPC=1 bash docker/build.sh 重出不带该层的镜像，再用本脚本换 IMG 起。"
if [ "$NVAPI" = "none" ]; then
  echo "  P-State：本次没挂库 ⇒ 电源档只能是 sleep（本次摆出的模板档已同步改掉，**已经建好的档要另改**：run/configure-v1.sh 那一题或 setfield power.mode sleep）。"
else
  echo "  P-State：本次挂的是 $NVAPI（默认 NVAPI=auto＝宿主自带那份；要换回包内那枚钉死副本就显式 NVAPI=package）。"
  echo "    它是否真生效不能只看文件在位——引擎起来后按 部署文档-v1.md 步骤 5 的判据核（空闲进 P8、负载回 16）。"
fi
