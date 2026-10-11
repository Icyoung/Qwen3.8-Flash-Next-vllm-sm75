#!/bin/bash
# configure-v1.sh —— 问几道题，把答案落到「面板档」与「容器参数」两处
#
# 为什么要有：官方模板与包内默认值锁的是基线（单人、256K、无 CPU KV）。多人用、内存小的机器、
# 想省电的机器，改的是同一批字段；让人去 JSON 里手改，改错一处（少个引号、KV 抬过头）表现不是报错，
# 而是"起得来但行为不对"。所以把取值范围、联动关系和门禁收进这一层。
#
# 规矩：回车＝**基线**（推荐值只写在题面上，不当默认值——否则低内存机器一回车就撞自己的门禁）；
#       序号与自定义值都允许；超过本机的值放行只提示，不拦；
#       只有低于能力线（砍上下文、砍单提示图数）与内存/显存算不过账才拦。
#
# 包 v1.2 在这一层加了两件事（文件名的 v1 指问答器这一代的形态，不随包版本递增）：
#   · G7 多一问"单条提示最多几张图"（以前两份包内档根本没写 --limit-mm-per-prompt，
#     对外默认一直吃引擎自带默认值，没人知道是多少）；
#   · G5 选 pstate 之前先查容器里那枚 P-State 库在不在——缺库而档是 pstate ⇒ 面板在启动引擎之前就抛，
#     少的是引擎不是省电（因果见 部署文档-v1.md 步骤 3）。
#
# 用法：
#   bash run/configure-v1.sh                      # 交互（推荐先 --dry-run 看一遍）
#   bash run/configure-v1.sh --dry-run            # 问答照跑，但一条都不写（拦写的那类门降级成 WARN）
#   YES=1 PROFILE=xxx bash run/configure-v1.sh    # 非交互：全部走基线（无人值守）
#   bash run/configure-v1.sh --show               # 只读：打印当前档的取值
#   bash run/configure-v1.sh --selftest           # 自检：调**生产同一份**门禁函数，含故意写坏的样张
set -u

NAME=${NAME:-sm75-017-console}
DATA=${DATA:-/var/lib/sm75-console}
CONFIG=${CONFIG:-$DATA/config.env}          # start-here-v2.sh 读它当默认值来源
TOOL=/console-data/tmp/console-edit-profile-v3.cjs
DRIVER=/console-data/tmp/console-driver-v1.cjs
PROFILES_DIR=/console-data/tmp/profiles     # start-here-v2.sh 把包内 run/profiles/*.json 放这儿
PROFILE=${PROFILE:-}

DRY=no SELFTEST=no SHOW=no INTERACTIVE=yes
[ -t 0 ] || INTERACTIVE=no
[ "${YES:-0}" = "1" ] && INTERACTIVE=no
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=yes ;;
    --selftest) SELFTEST=yes ;;
    --show) SHOW=yes ;;
    --yes) INTERACTIVE=no ;;
    *) echo "用法：bash run/configure-v1.sh [--dry-run|--show|--selftest]"; exit 2 ;;
  esac
done

GiB=1073741824
note() { printf '%s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; }
# 拦下来的是"真写"这一步：--dry-run 只预览，所以这类门在预演里降级成 WARN（与 start-here 同一套语义）
stop() { if [ "$DRY" = "yes" ]; then warn "$*"; else printf 'BLOCK %s\n' "$*"; exit 4; fi; }
is_int() { printf '%s' "$1" | grep -qE '^[0-9]+$'; }

# ---------- 门禁函数：生产与 --selftest 共用这一份 ----------
# 之前自检里复制了第二套 awk，把生产那处改坏自检照样绿＝半假绿。现在只有一处定义。
gate_cpu_kv() { awk -v a="$1" -v w="$2" 'BEGIN{exit !(a>=w+8)}'; }        # $1=可用 GiB $2=想开 GiB
gate_mem_cap() { awk -v m="$1" -v r="$2" 'BEGIN{exit !(m>=r)}'; }         # 容器上限不得大于宿主总量
gate_mem_room() { awk -v m="$1" -v r="$2" 'BEGIN{exit !(m>=r+8)}'; }      # 距宿主总量不足 8 GiB 只 WARN
gate_util() { awk -v f="$1" -v t="$2" -v u="$3" 'BEGIN{ if (t > 0 && (f / t) >= u) exit 0; else exit 1 }'; }

if [ "$SELFTEST" = "yes" ]; then
  rc=0
  gate_cpu_kv 97 32 || { echo "  SELFTEST_FAIL 可用 97G 配 32G 被误拦"; rc=1; }
  gate_cpu_kv 20 32 && { echo "  SELFTEST_FAIL 可用 20G 配 32G 该拦没拦"; rc=1; }
  gate_cpu_kv 8 4 && { echo "  SELFTEST_FAIL 可用 8G 配 4G 该拦没拦"; rc=1; }
  gate_mem_cap 251 264 && { echo "  SELFTEST_FAIL 容器上限 264 > 宿主 251 该拦没拦"; rc=1; }
  gate_mem_cap 251 236 || { echo "  SELFTEST_FAIL 容器上限 236 ≤ 宿主 251 被误拦"; rc=1; }
  gate_mem_room 240 236 && { echo "  SELFTEST_FAIL 只差 4 GiB 该 WARN 却没判出来"; rc=1; }
  [ "$(gate_util 14620 15933 0.90; echo $?)" = "0" ] || { echo "  SELFTEST_FAIL util 0.90 在上限 0.9176 内却被拒"; rc=1; }
  [ "$(gate_util 14620 15933 0.92; echo $?)" = "1" ] || { echo "  SELFTEST_FAIL util 0.92 高于上限 0.9176 却没红"; rc=1; }
  [ "$(gate_util 14620 0 0.92; echo $?)" = "1" ] || { echo "  SELFTEST_FAIL 总显存读成 0（除零）却没拒"; rc=1; }
  # 样张：故意喂坏输入，校验器必须判非数字
  is_int "8x" && { echo "  SELFTEST_FAIL is_int 放过了 8x"; rc=1; }
  is_int "" && { echo "  SELFTEST_FAIL is_int 放过了空串"; rc=1; }
  [ "$rc" = "0" ] && { echo "SELFTEST_DONE 四道门禁的正/负控与坏样张各归各位"; exit 0; } || exit 6
fi

# ---------- 门禁（全部只读；排在自家任何写盘之前） ----------
docker inspect "$NAME" >/dev/null 2>&1 || stop "容器不在册：$NAME（先跑 run/start-here-v2.sh）"
[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = "true" ] || stop "容器没在跑：$NAME"
docker exec "$NAME" test -s "$TOOL" || stop "容器里看不到 $TOOL（start-here 没拷进去？包内 tools/panel/ 有）"

ids=$(docker exec "$NAME" node "$TOOL" - list 2>/dev/null) || stop "列档失败（面板没响应？）"
printf '%s' "$ids" | grep -q '^personal-' || stop "容器里还没有任何档：先跑 docker exec $NAME node $DRIVER setup"
if [ -z "$PROFILE" ]; then
  n=$(printf '%s\n' "$ids" | grep -c '^personal-')
  if [ "$n" = "1" ] && [ "$INTERACTIVE" = "no" ]; then
    PROFILE=$(printf '%s\n' "$ids" | awk -F'\t' '/^personal-/{print $1; exit}')
    note "  自动选中唯一的档：$PROFILE"
  elif [ "$INTERACTIVE" = "yes" ]; then
    echo "  ? 要配置哪一档？"
    i=1; declare -a PID=()
    while IFS=$'\t' read -r pid pname plen; do
      case "$pid" in personal-*) printf '      %s) %s  (%s 项参数)\n' "$i" "$pname" "$plen"; PID[$i]="$pid"; i=$((i+1));; esac
    done <<< "$ids"
    printf '  选择：'; IFS= read -r ans
    [ -n "${PID[${ans:-0}]:-}" ] || stop "没这个序号"
    PROFILE="${PID[$ans]}"
  else
    stop "容器里有 $n 档，非交互模式必须显式 PROFILE=<id>（列档：docker exec $NAME node $TOOL - list）"
  fi
fi

dump=$(docker exec "$NAME" node "$TOOL" "$PROFILE" dump 2>/dev/null) || stop "读不到档 $PROFILE"
[ -n "$dump" ] || stop "档 $PROFILE 读回来是空的（面板 API 异常，别往下走）"
cur_flag() { printf '%s\n' "$dump" | awk -F'\t' -v f="$1" '$1=="FLAG"&&$2==f{print $3; exit}'; }
cur_field() { printf '%s\n' "$dump" | awk -F'\t' -v k="$1" '$1==k{print $2; exit}'; }
cur_env() { printf '%s\n' "$dump" | awk -F'\t' -v k="$1" '$1=="ENV"&&$2==k{print $3; exit}'; }
cur_argv0() { printf '%s\n' "$dump" | awk -F'\t' '$1=="ARGV0"{print $2; exit}'; }

if [ "$SHOW" = "yes" ]; then
  echo "=== $PROFILE 当前取值（只读）==="
  printf '%s\n' "$dump"
  echo "SHOW_DONE"
  exit 0
fi

# ---------- 宿主实测（内存/显存门用它，不用推算） ----------
# /proc/meminfo 只有 Linux 有；非 Linux 的开发机（或台架）可以用 SM75_MEM="总 可用" 显式给两个 GiB 数。
# 给了就用、不给且读不到就 BLOCK——**不拿默认值兜底**，否则"读不到"会变成"按 0 放行"或"按猜的放行"。
mt=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo 2>/dev/null)
ma=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo 2>/dev/null)
if [ -n "${SM75_MEM:-}" ]; then
  set -- $SM75_MEM
  mt="$1"; ma="${2:-$1}"
  note "  用 SM75_MEM 覆盖宿主内存读数：总 ${mt} GiB / 可用 ${ma} GiB"
fi
[ -n "$mt" ] || stop "读不到 /proc/meminfo 的 MemTotal（非 Linux 就显式给 SM75_MEM=\"总 可用\"）"
[ -n "$ma" ] || stop "读不到 /proc/meminfo 的 MemAvailable（非 Linux 就显式给 SM75_MEM=\"总 可用\"）"
gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1 | tr -dc '0-9')
gpu_tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1 | tr -dc '0-9')
note "  实测：宿主内存 总 ${mt} GiB / 可用 ${ma} GiB；最小那张卡 空闲 ${gpu_free:-读不到} / 共 ${gpu_tot:-读不到} MiB"

# ---------- 问答 ----------
declare -a PLAN=()      # 面板档：每条形如 "mode|arg1|arg2"
declare -a CFG=()       # 容器层：写进 config.env 的 KEY=VALUE
add_plan() { PLAN+=("$1"); }
add_cfg() { CFG+=("$1"); }

# 选项一律"值=说明"。输入先按**字面值**匹配，命中不上的 1..N 才当序号——
# 否则并发那题（选项值 4/8/16）里输入"4"会被当成第 4 个选项，静默串位。
# pick <问题> <基线值> <当前档值> 选项…
#   非交互 ⇒ 一律取**基线值**（不是"当前值"！否则无人值守会把别人改过的值当默认保留）
#   交互   ⇒ 回车取当前值（＝"这一题不改"），题面同时把基线值标出来
pick() {
  local q="$1" base="$2" cur="$3"; shift 3
  local ans o i=1
  if [ "$INTERACTIVE" != "yes" ]; then CUR="$base"; note "  · $q → $CUR（非交互＝基线）"; return 0; fi
  printf '  ? %s\n' "$q"
  printf '      基线＝%s，当前档＝%s\n' "$base" "$cur"
  for o in "$@"; do printf '      %s) %s\n' "$i" "$o"; i=$((i+1)); done
  printf '      或直接输入自定义值（回车＝不改，保持 %s）\n' "$cur"
  printf '  选择：'; IFS= read -r ans || ans=""
  if [ -z "$ans" ]; then CUR="$cur"; return 0; fi
  for o in "$@"; do
    [ "${o%%=*}" = "$ans" ] && { CUR="$ans"; note "      ⇒ $CUR"; return 0; }
  done
  if printf '%s' "$ans" | grep -qE '^[1-9][0-9]*$' && [ "$ans" -le "$#" ]; then
    CUR=$(printf '%s' "${@:$ans:1}" | cut -d'=' -f1); note "      ⇒ $CUR"; return 0
  fi
  CUR="$ans"; note "      ⇒ 自定义 $CUR"
}
confirm() {
  [ "$INTERACTIVE" = "yes" ] || return 0
  printf '  ! %s\n  确认继续？(y/N)：' "$1"; IFS= read -r a; [ "$a" = "y" ] || return 1; return 0
}

# G0 要不要另建一档（包内第二靶子）
if [ "$INTERACTIVE" = "yes" ]; then
  pick "要不要从包内 JSON 另建一档（例如第二靶子 AutoRound）？" "0" "0" \
    "0=不建，就配当前这一档" "1=建"
  if [ "$CUR" = "1" ]; then
    docker exec "$NAME" test -d "$PROFILES_DIR" || stop "容器里没有 $PROFILES_DIR（start-here-v2 没拷 profile？重跑它）"
    pick "用包内哪份 JSON？" "flash-next-tp8-256k-autoround.json" "flash-next-tp8-256k-autoround.json" \
      "flash-next-tp8-256k-autoround.json=AutoRound W4A16（内存要求高，见 README 基础条件）" \
      "flash-next-tp8-256k-nomtp.json=FP8PLE（官方 tested 修订）"
    PF="$CUR"
    pick "新档 id（只能小写字母、数字与连字符，最长 48；这是上游删除路由的字符集，带 . 或 _ 的档将来删不掉）？" "personal-autoround-1" "personal-autoround-1"
    NEWID="$CUR"
    # 字符集与长度对齐上游 profile-delete.mjs:22 的 /^[a-z0-9][a-z0-9-]{0,47}$/：
    # 这里放行 . 或 _ 的话，档建得出来、但那条原生删除路由的正则匹配不到它＝永远删不掉的孤档。
    printf '%s' "$NEWID" | grep -qE '^[a-z0-9][a-z0-9-]{0,47}$' || stop "新档 id 不合规（要匹配上游删除路由的字符集：小写字母/数字/连字符，48 字符内）：$NEWID"
    # 包内档的 args[0] 是模板占位符 ${MODEL}，clone 时必须给真路径（否则建出来起不来）
    pick "新档用哪个模型目录（容器内绝对路径）？" "/models" "/models"
    MDL0="$CUR"
    printf '%s' "$MDL0" | grep -q '^/' || stop "模型目录必须是绝对路径：$MDL0"
    echo "--- clone"
    docker exec "$NAME" node "$TOOL" - clone "$NEWID" "由 configure 建的档" "$PROFILES_DIR/$PF" "$MDL0" \
      || stop "clone 失败（目标 id 可能已在位；面板 POST 是 upsert，脚本不会替你覆盖别人的档）"
    PROFILE="$NEWID"
    dump=$(docker exec "$NAME" node "$TOOL" "$PROFILE" dump 2>/dev/null) || stop "新档回读不到"
    note "  已切到新档 $PROFILE 继续配置"
  fi
fi

# G1 并发
seq_cur=$(cur_flag --max-num-seqs); [ -n "$seq_cur" ] || seq_cur=4
pick "几个人同时用？（并发上限＝同时处理的请求数）" 4 "$seq_cur" \
  "4=1 人（基线）" "8=2–4 人" "16=5 人以上"
SEQ="$CUR"
is_int "$SEQ" || stop "--max-num-seqs 要整数（填了 $SEQ）"
MMBT=4096; CG='{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4]}'
case "$SEQ" in
  4) : ;;
  8) MMBT=8192; CG='{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}' ;;
  16) MMBT=8192; CG='{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16]}' ;;
  *) warn "自定义并发 $SEQ：批量与图捕获按基线配，要另调就再跑一次" ;;
esac
cg_cur=$(cur_flag --compilation-config)
if [ "$SEQ" != "$seq_cur" ]; then
  add_plan "setarg|--max-num-seqs|$SEQ"
  add_plan "setarg|--max-num-batched-tokens|$MMBT"
  # 图捕获尺寸必须跟着并发走：并发抬到 8/16 而捕获仍是 [1,2,4] ⇒ 大 batch 落不到 FULL 图，
  # 表现是"起得来但每步更慢"。这一条以前算出来没落地。
  [ -n "$cg_cur" ] && [ "$cg_cur" != "$CG" ] && add_plan "setarg|--compilation-config|$CG"
  [ -z "$cg_cur" ] && add_plan "addarg|--compilation-config|$CG"
fi

# G2 上下文
ctx_cur=$(cur_flag --max-model-len); [ -n "$ctx_cur" ] || ctx_cur=262144
pick "上下文长度上限？" 262144 "$ctx_cur" \
  "262144=256K（基线，长文能力完整）" "131072=128K" "32768=32K" "8192=8K"
CTX="$CUR"
is_int "$CTX" || stop "--max-model-len 要整数（填了 $CTX）"
if [ "$CTX" != "$ctx_cur" ] && [ "$CTX" -lt "$ctx_cur" ] 2>/dev/null; then
  confirm "把上下文从 $ctx_cur 降到 $CTX＝关掉这部分长文能力（不是性能项，是能力项）" || CTX="$ctx_cur"
fi
[ "$CTX" != "$ctx_cur" ] && add_plan "setarg|--max-model-len|$CTX"

# G3 CPU KV：回车＝基线（不开），推荐值只写在题面
kvt_cur=$(cur_flag --kv-transfer-config)
kv_max=$((ma - 8)); [ "$kv_max" -lt 0 ] && kv_max=0
kv_rec=0
for t in 4 8 16 32; do [ "$kv_max" -ge "$t" ] && kv_rec=$t; done
kv_now=0
if [ -n "$kvt_cur" ]; then
  kb=$(printf '%s' "$kvt_cur" | tr -d ' ' | sed -n 's/.*"cpu_bytes_to_use":\([0-9]*\).*/\1/p')
  [ -n "$kb" ] && kv_now=$((kb / GiB))
fi
pick "CPU KV 开多大？（把一部分 KV 缓存挪进内存，多人/长文更稳；本机实测可用 ${ma} GiB，最多约 ${kv_max} GiB，推荐 ${kv_rec}）" \
  "0" "$kv_now" "4=4 GiB" "8=8 GiB" "16=16 GiB" "32=32 GiB" "0=不开（基线）"
KV="$CUR"
is_int "$KV" || stop "CPU KV 要填整数 GiB（$KV 不是）"
KVT=''
if [ "$KV" != "0" ]; then
  gate_cpu_kv "$ma" "$KV" || stop "这台机实测可用内存 ${ma} GiB，扣 8 GiB 留量后放不下 ${KV} GiB CPU KV（最多约 $kv_max GiB）"
  [ "$KV" -gt 32 ] && warn "自定义 ${KV} GiB 超出包内验证过的 4–32 区间，按你填的走"
  KVT=$(printf '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"CPUOffloadingSpec","cpu_bytes_to_use":%d}}' $((KV * GiB)))
  if [ -n "$kvt_cur" ]; then
    [ "$kvt_cur" != "$KVT" ] && add_plan "setarg|--kv-transfer-config|$KVT"
  else
    add_plan "addarg|--kv-transfer-config|$KVT"
  fi
else
  [ -n "$kvt_cur" ] && add_plan "delarg|--kv-transfer-config"
fi

# G4 容器内存上限（靶子基线＋CPU KV）
mem_bytes=$(docker inspect -f '{{.HostConfig.Memory}}' "$NAME" 2>/dev/null | tr -dc '0-9')
mem_g=$(( ${mem_bytes:-0} / GiB ))
base_def=112; [ "$mem_g" -ge 200 ] && base_def=232
pick "这台机跑的靶子（权重档）？决定容器内存上限基线（当前容器实读上限 ${mem_g:-0} GiB）" 112 "$base_def" \
  "112=FP8PLE（官方 tested 修订，权重加载峰值实测 110.7 GiB）" \
  "232=AutoRound W4A16（实测容器内存占用 231.4 GiB）"
BASE_MEM="$CUR"
is_int "$BASE_MEM" || stop "靶子基线要整数 GiB（$BASE_MEM 不是）"
MEM=$((BASE_MEM + KV))
gate_mem_cap "$mt" "$MEM" || stop "容器内存上限 ${MEM} GiB（基线 $BASE_MEM＋CPU KV ${KV}）大于宿主总内存 ${mt} GiB——降 CPU KV 或换靶子"
gate_mem_room "$mt" "$MEM" || warn "容器上限 ${MEM} GiB 距宿主总内存不足 8 GiB，宿主侧周转可能吃紧"
add_cfg "MEM_LIMIT=$((MEM * GiB))"

# G5 电源
mode_cur=$(cur_field FIELD_power_mode); [ -n "$mode_cur" ] || mode_cur=pstate
pick "电源托管方式？" pstate "$mode_cur" "pstate=常驻 P-State（基线，需要 NVAPI 库）" "sleep=空闲自动休眠（不需要 NVAPI，但唤醒更慢）"
MODEV="$CUR"
case "$MODEV" in pstate|sleep) : ;; *) stop "power.mode 只能是 pstate 或 sleep（填了 $MODEV）" ;; esac
# 选 pstate 之前先证容器里真有那枚库。这不是"少个省电功能"的取舍：控制台 pstate 分支里
# 那句"库不存在就抛"排在**启动引擎之前**（standalone.mjs 进分支 → existsSync 不过 → throw，
# 再往后才 spawn 监督进程、才 launch vllm），所以缺库还留 pstate 的结果是引擎根本起不来。
# 查的是面板自己判的那条路径，不是"我 --volume 过"。
if [ "$MODEV" = "pstate" ]; then
  if docker exec "$NAME" test -e /usr/local/nvidia/lib64/libnvidia-api.so.1 2>/dev/null; then
    note "  · pstate 前置检查：容器内 /usr/local/nvidia/lib64/libnvidia-api.so.1 在位"
  else
    stop "容器里没有 /usr/local/nvidia/lib64/libnvidia-api.so.1 ⇒ 电源档只能选 sleep。要按 pstate 用，先用带库的方式重建容器（run/start-here-v2.sh 默认 NVAPI=auto 探宿主自带那份；宿主没有就按 docker/NVAPI-获取说明-v1.md 从你自己的驱动取件）"
    # stop() 在 --dry-run 里只 WARN；那时流程还要往下走，所以这里把选择**就地降成 sleep**，
    # 免得预演列出一条真跑必然非法的 setfield|power.mode|pstate 计划。
    printf -v MODEV '%s' sleep   # 用 printf -v 而不是单独一行 MODEV=sleep：后者会被 shellcheck 读成"像命令"（SC2209）
    note "      ⇒ 本次按 sleep 继续（真跑这一题会直接 BLOCK）"
  fi
fi
[ "$MODEV" != "$mode_cur" ] && add_plan "setfield|power.mode|$MODEV"
P8=''
if [ "$MODEV" = "pstate" ]; then
  p8_cur=$(cur_field FIELD_power_idleSeconds); [ -n "$p8_cur" ] || p8_cur=120
  pick "空闲多久进 P8（秒）？" 120 "$p8_cur" "120=2 分钟（本包出厂默认）" "1800=30 分钟" "86400=24 小时（测速常驻）"
  P8="$CUR"
  is_int "$P8" || stop "P8 秒数要非负整数（$P8 不是）"
  [ "$P8" != "$p8_cur" ] && add_plan "setfield|power.idleSeconds|$P8"
  add_cfg "PSTATE_IDLE_TIMEOUT=$P8"
fi

# G6 权重目录与模型名
# args[0] 可能是模板占位符 ${MODEL}（官方模板就这么存）。占位符不是路径：拿它当默认值再走
# "必须绝对路径"的自检，等于让建档流程自己 BLOCK 自己——这一题在占位符状态下跳过，
# 真路径要在 clone（G0）或面板 setup 那一步给。
mdl_cur=$(cur_argv0)
srv_cur=$(cur_flag --served-model-name); [ -n "$srv_cur" ] || srv_cur="Flash-Next-TP8"
case "$mdl_cur" in
  /*) pick "模型目录（容器内路径）？" "$mdl_cur" "$mdl_cur" "/models=权重挂在 /models（包内默认挂法）"
      MDL="$CUR"
      printf '%s' "$MDL" | grep -q '^/' || stop "模型目录必须是绝对路径：$MDL"
      [ "$MDL" != "$mdl_cur" ] && add_plan "setmodel|$MDL|-" ;;
  *)  note "  · 模型目录当前是模板占位符 ${mdl_cur:-空}（建档时没替换）⇒ 这一题跳过；要换真路径就走 G0 的 clone 或面板 setup" ;;
esac
pick "API 里调用用的模型名（served-model-name）？" "$srv_cur" "$srv_cur"
[ "$CUR" != "$srv_cur" ] && add_plan "setarg|--served-model-name|$CUR"

# G7 能力开关
ipc_cur=$(cur_env VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC); [ -n "$ipc_cur" ] || ipc_cur=1
pick "PCIe-IPC all-reduce（这台机上 decode ＋30% 的那一项，见 README §1 与 §5）？" 1 "$ipc_cur" \
  "1=开（推荐，需要镜像带回填层）" "0=关（官方形态，后端退回 FIREFLY_AR）"
IPC="$CUR"
case "$IPC" in 0|1) : ;; *) stop "这一项只能填 0 或 1（填了 $IPC）" ;; esac
[ "$IPC" != "$ipc_cur" ] && add_plan "setenv|VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC|$IPC"
hc_cur=$(cur_env VLLM_SM75_QWEN38_HC_GEMV); [ -n "$hc_cur" ] || hc_cur=1
pick "HC_GEMV 内核？（当前档的值 $hc_cur）" "$hc_cur" "1=开（FP8PLE 靶的档值）" "0=关（AutoRound 靶的档值）"
HC="$CUR"
case "$HC" in 0|1) : ;; *) stop "这一项只能填 0 或 1（填了 $HC）" ;; esac
[ "$HC" != "$hc_cur" ] && add_plan "setenv|VLLM_SM75_QWEN38_HC_GEMV|$HC"
mtp_cur=$(cur_flag --speculative-config)
mtp_def=0; [ -n "$mtp_cur" ] && mtp_def=3
pick "MTP 投机解码？" 0 "$mtp_def" \
  "0=关（基线；这台机上开 MTP3 实测是负优化，数据不在本包内）" "3=开 MTP3（自担风险）"
MTP="$CUR"
case "$MTP" in 0|3) : ;; *) stop "这一题只能填 0 或 3（填了 $MTP）" ;; esac
if [ "$MTP" = "3" ] && [ -z "$mtp_cur" ]; then
  add_plan 'addarg|--speculative-config|{"method":"mtp","num_speculative_tokens":3}'
elif [ "$MTP" = "0" ] && [ -n "$mtp_cur" ]; then add_plan "delarg|--speculative-config"; fi

# G7·单条提示的图数（多模态能力上限）。包 v1.1 及以前两份档**根本没写这一条** ⇒ 对外默认一直吃
# 引擎自带默认值、没人知道是多少；v1.2 起显式写死，基线 16。
# 这台机实测（1024×1024 PNG，TP8／256K／kv-cache-memory-bytes 2415919104，见 部署文档-v1.md 的多图那一节）：
#   4 张 1.6 s／16 张 4.7 s／32 张 15.8 s／33 张那格被引擎以 400 拒（原因本机未取证）。
# 单张 1024×1024＝1024 个 vision token；编码器缓存预算恒为 16384 tokens、**不随这一条变**。
# 超预算**跨张**是分配时逐出⇒分波编码、代价是时延（2 倍图数≈3.4 倍时间）；
# 真正的硬墙在**单张尺寸**：单张 item 的 embed 数超过缓存上限是运行期 400。判据与行号见 部署文档-v1.md 步骤 4·多图。
mm_cur=$(cur_flag --limit-mm-per-prompt)
mm_img=""; mm_flat=""
if [ -n "$mm_cur" ]; then
  mm_flat=$(printf '%s' "$mm_cur" | tr -d ' ')
  mm_img=$(printf '%s' "$mm_flat" | sed -n 's/.*"image":\([0-9]*\).*/\1/p')
  # 档里已有这一条却解析不出 image ⇒ 别猜、别整串覆盖：停下来让人看一眼现值。
  [ -n "$mm_img" ] || stop "档里 --limit-mm-per-prompt 的值读不出 image 字段（$mm_cur）。请先看 docker exec $NAME node $TOOL $PROFILE dump 再手工处理这一条"
fi
[ -n "$mm_img" ] && mm_now="$mm_img" || mm_now=16      # ABSENT 时题面按基线 16 起（这一题的目的就是把这一条写死）
pick "单条提示最多几张图？（能力上限不是性能项。这台机实测：4 张 1.6 s／16 张 4.7 s／32 张 15.8 s，33 张被 400 拒；编码器缓存预算恒为 16384 tokens，超了是分波编码、代价是时延）" \
  16 "$mm_now" "16=16 张（基线）" "4=4 张" "32=32 张（超编码器缓存预算，首包时延约 3.4 倍）" "0=整条关掉图像输入"
MM="$CUR"
is_int "$MM" || stop "--limit-mm-per-prompt 的 image 要整数（填了 $MM）"
if [ "$MM" != "0" ] && [ "$MM" -lt 4 ]; then
  stop "单条提示图数填 $MM＝砍多模态能力：本包只在 4／16／32 三档上实测过，低于 4 视为砍能力（照这一层既有纪律：低于能力线才拦）。要少图填 4，整条不要图像填 0。"
fi
[ "$MM" -gt 32 ] && warn "自定义 $MM 张超出实测范围（本机只核到 32；33 那一格是被 400 拒的，原因未取证），按你填的走"
[ "$MM" -gt 16 ] && warn "单提示图数 $MM 超出编码器缓存预算 16384 tokens（16 张 1024×1024 正好用完）⇒ 分波编码，首包时延明显上涨"
if [ -n "$mm_img" ] && [ "$MM" != "$mm_img" ] && [ "$MM" -lt "$mm_img" ] 2>/dev/null; then
  confirm "把单条提示图数从 $mm_img 降到 $MM＝关掉这部分多模态能力（那是能力项不是性能项）" || MM="$mm_img"
fi
# 落盘形态：**只替换 image 那一个数字**，其余键（video／audio／以后新增的）原样留着。
# 整段重画会替人凭空补上 "video":0（＝静默把视频能力关掉），与"只改声明的那一处"的纪律相冲。
if [ -n "$mm_cur" ]; then
  MMJSON=$(printf '%s' "$mm_flat" | sed -e "s/\"image\":[0-9][0-9]*/\"image\":$MM/")
else
  MMJSON=$(printf '{"image":%s,"video":0}' "$MM")
fi
# 比的是**去掉空白后的现值**：面板与人手工存的 JSON 可能带空格（"image": 4），
# 拿原样串比会因空白形态不同而多发一条内容没变的 setarg。
if [ "$MMJSON" != "$mm_flat" ]; then
  if [ -n "$mm_cur" ]; then add_plan "setarg|--limit-mm-per-prompt|$MMJSON"; else add_plan "addarg|--limit-mm-per-prompt|$MMJSON"; fi
fi

# G8 容器端口
pick "面板对外端口？" 1615 "${PANEL_PORT:-1615}"
is_int "$CUR" || stop "面板端口要整数（$CUR 不是）"
add_cfg "PANEL_PORT=$CUR"
pick "引擎 API 对外端口？" 8000 "${API_PORT:-8000}"
is_int "$CUR" || stop "API 端口要整数（$CUR 不是）"
add_cfg "API_PORT=$CUR"
pick "只绑回环？（面板与引擎都不带鉴权，局域网可见＝谁都能用）" 127.0.0.1 "127.0.0.1" \
  "127.0.0.1=只本机（基线）" "0.0.0.0=局域网可见（自行加 api-key 与边界控制）"
add_cfg "BIND_HOST=$CUR"
[ "$CUR" = "0.0.0.0" ] && warn "选了 0.0.0.0：请务必给引擎加 --api-key，并自己保证网络边界"

# ---------- 预览与确认 ----------
echo "=== 将要改的档内参数（$PROFILE）==="
if [ "${#PLAN[@]}" = "0" ]; then echo "  （无：当前取值已经等于你的选择，一条都不会写）"; fi
for p in "${PLAN[@]}"; do echo "  · $(printf '%s' "$p" | cut -d'|' -f1) $(printf '%s' "$p" | cut -d'|' -f2,3 | tr '|' ' ')"; done
echo "=== 将要写进 $CONFIG（容器层，start-here-v2.sh 读）==="
for c in "${CFG[@]}"; do echo "  · $c"; done
echo "=== 显存门 ==="
if [ -n "${gpu_tot:-}" ] && [ "${gpu_tot:-0}" != "0" ]; then
  util_cur=$(cur_flag --gpu-memory-utilization); [ -n "$util_cur" ] || util_cur=0.92
  if gate_util "${gpu_free:-0}" "$gpu_tot" "$util_cur"; then
    note "  此刻单卡 util 上限≈$(awk -v f="$gpu_free" -v t="$gpu_tot" 'BEGIN{printf "%.4f", f/t}')，档里 $util_cur 在其内"
  else
    warn "档里 util=$util_cur 高于此刻上限 $(awk -v f="$gpu_free" -v t="$gpu_tot" 'BEGIN{printf "%.4f", f/t}')——引擎会在启动时拒，先降档或清卡"
  fi
fi
if [ "${#PLAN[@]}" != "0" ] && [ "$DRY" != "yes" ] && [ "$INTERACTIVE" = "yes" ]; then
  printf '  确认执行以上改动？(y/N)：'; IFS= read -r go
  [ "$go" = "y" ] || { echo "CANCELLED 一条都没写（档未改、config.env 未写）"; exit 0; }
fi
if [ "$DRY" = "yes" ]; then
  echo "DRY_RUN_OK 上面就是将要执行的改动。本次没有改档、没有写 $CONFIG。"
  exit 0
fi

# ---------- 落盘：先容器层文件，再档（档侧每条自带回读断言与回滚） ----------
mkdir -p "$DATA" || { echo "WRITE_FAIL 建不出 $DATA"; exit 3; }
touch "$CONFIG" || { echo "WRITE_FAIL 写不出 $CONFIG"; exit 3; }
chmod 600 "$CONFIG" 2>/dev/null
got_perm=$(stat -c '%a' "$CONFIG" 2>/dev/null || stat -f '%Lp' "$CONFIG" 2>/dev/null || echo 未知)
[ "$got_perm" = "600" ] || warn "config.env 权限是 $got_perm 而不是 600（这台机 stat 读不到？自己 chmod 600 $CONFIG）"
for c in "${CFG[@]}"; do
  k=${c%%=*}
  # 用 awk 原地替换而不是 sed：值里出现 & | / 时 sed 会把替换式改坏或静默改错
  awk -v key="$k" -v line="$c" 'BEGIN{FS=OFS="="; done=0}
    { if ($0 ~ "^"key"=") { if (!done) { print line; done=1 } ; next } print }
    END{ if (!done) print line }' "$CONFIG" > "$CONFIG.tmp" || { echo "WRITE_FAIL awk 处理 $k 失败"; rm -f "$CONFIG.tmp"; exit 3; }
  mv "$CONFIG.tmp" "$CONFIG" || { echo "WRITE_FAIL 换不回 config.env"; exit 3; }
  grep -qx "$c" "$CONFIG" || { echo "ASSERT_FAIL config.env 回读不到 $c"; exit 7; }
  n=$(grep -c "^$k=" "$CONFIG"); [ "$n" = "1" ] || { echo "ASSERT_FAIL $k 在 config.env 里有 $n 行（应恰好 1）"; exit 7; }
done
note "  config.env 就绪（$(grep -c '^' "$CONFIG") 行、权限 $got_perm）"

for p in "${PLAN[@]}"; do
  m=$(printf '%s' "$p" | cut -d'|' -f1); x=$(printf '%s' "$p" | cut -d'|' -f2); y=$(printf '%s' "$p" | cut -d'|' -f3-)
  echo "--- $m $x"
  if [ "$m" = "setmodel" ]; then
    docker exec "$NAME" node "$TOOL" "$PROFILE" setmodel "$x" \
      || stop "setmodel 失败，后面的改动没做（档可能已部分变更，用 --show 核对）"
  elif [ "$m" = "delarg" ] || [ "$m" = "delswitch" ]; then
    docker exec "$NAME" node "$TOOL" "$PROFILE" "$m" "$x" \
      || stop "$m $x 失败，后面的改动没做（v2 自带回滚，用 --show 核对）"
  else
    docker exec "$NAME" node "$TOOL" "$PROFILE" "$m" "$x" "$y" \
      || stop "$m $x 失败，后面的改动没做（v2 自带回滚，用 --show 核对）"
  fi
done

# ---------- 终态自检：断言产物，不是"我发过命令" ----------
echo "=== 终态自检 ==="
dump2=$(docker exec "$NAME" node "$TOOL" "$PROFILE" dump 2>/dev/null) || { echo "ASSERT_FAIL 回读失败"; exit 7; }
check_fail=no
chk() { if [ "$2" = "$3" ]; then echo "  PASS $1 = $3"; else echo "  ASSERT_FAIL $1 期望 $2 实为 ${3:-ABSENT}"; check_fail=yes; fi; }
g_flag() { printf '%s\n' "$dump2" | awk -F'\t' -v f="$1" '$1=="FLAG"&&$2==f{print $3; exit}'; }
g_field() { printf '%s\n' "$dump2" | awk -F'\t' -v k="$1" '$1==k{print $2; exit}'; }
g_env() { printf '%s\n' "$dump2" | awk -F'\t' -v k="$1" '$1=="ENV"&&$2==k{print $3; exit}'; }
chk "--max-num-seqs" "$SEQ" "$(g_flag --max-num-seqs)"
chk "--max-model-len" "$CTX" "$(g_flag --max-model-len)"
# --limit-mm-per-prompt 断的是"解析出来的 image 数值等于要求的数"，不做整串比对：
# 面板对 JSON 值会做对象/字符串往返（--compilation-config 走的就是同一条通道），整串比会因空白形态假红。
# 但"回读不到数值"本身算失败——否则这一条被面板整段吞掉时 chk 会拿空串比空串、报绿。
got_mm=$(printf '%s' "$(g_flag --limit-mm-per-prompt)" | tr -d ' ' | sed -n 's/.*"image":\([0-9]*\).*/\1/p')
if [ -z "$got_mm" ]; then
  echo "  ASSERT_FAIL --limit-mm-per-prompt 回读不到 image 数值（实读：$(g_flag --limit-mm-per-prompt)）"; check_fail=yes
else
  chk "--limit-mm-per-prompt 的 image" "$MM" "$got_mm"
fi
chk "power.mode" "$MODEV" "$(g_field FIELD_power_mode)"
[ "$MODEV" = "pstate" ] && chk "power.idleSeconds" "$P8" "$(g_field FIELD_power_idleSeconds)"
if [ "$KV" = "0" ]; then
  [ -z "$(g_flag --kv-transfer-config)" ] && echo "  PASS --kv-transfer-config 不在位（按选择关闭）" \
    || { echo "  ASSERT_FAIL 说要关 CPU KV 但档里还在"; check_fail=yes; }
elif [ -n "$KVT" ]; then
  chk "CPU KV 字节数" "$KVT" "$(g_flag --kv-transfer-config)"
fi
[ "$IPC" = "1" ] && { [ "$(g_env VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC)" = "1" ] \
  && echo "  PASS PCIe-IPC 开关在位" || { echo "  ASSERT_FAIL PCIe-IPC 没落进 env"; check_fail=yes; }; }
if [ "$check_fail" = "yes" ]; then echo "DONE_WITH_FAILS 上面每条都要解决或显式撤回；引擎尚未启动"; exit 7; fi
echo "CONFIGURE_DONE 档与 config.env 都已按选择落好"
echo "  容器层改动（内存上限、端口、绑定、PSTATE_IDLE_TIMEOUT）要重建容器才生效："
echo "    DATA=$DATA bash run/start-here-v2.sh   （同名容器需先 docker rm $NAME）"
echo "  起引擎：docker exec $NAME node $DRIVER start $PROFILE　　核对：docker exec $NAME node $TOOL $PROFILE dump"
