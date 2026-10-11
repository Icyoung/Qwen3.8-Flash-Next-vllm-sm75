#!/bin/bash
# build.sh —— 0.1.7 自包含部署包的镜像构建编排（唯一构建入口）
#
# 三层模型（每层独立标签，任一层可跳）：
#   层 0  官方 0.1.7-beta2 ultra      vllm-sm75:v0.1.7-ultra-beta2          （由官方 docker/build.sh 产出，或你已有）
#   层 1  GPU 探测超时 3s→20s       …-to20s                                （本机 hardware.py 实测 6.0-6.9s，不修面板拒启）
#   层 2  FlashInfer PCIe-IPC 回填  …-to20s-pcieipc                         （默认推荐；SKIP_PCIEIPC=1 退回层 1）
#   NVAPI **不在任何层里**：0.1.7 的形态是宿主文件只读 bind 进容器（见 NVAPI-获取说明-v1.md）。
#   包 v1.2 起，bind 哪一份由 run/start-here-v2.sh 决定（默认探宿主驱动自带那份）；本脚本这段只查**包内物料**是否完好，
#   缺它不再阻断构建——构建从头到尾不碰这个库（0.1.6 那种"烘进镜像"的形态在 0.1.7 已不适用）。
#
# 用法：
#   bash build.sh                      # 层0 若已在位则跳过，然后层1、层2
#   MODE=layers bash build.sh          # 只做层1+层2（要求层0 标签已在位）
#   SKIP_TO20S=1 bash build.sh         # 只做层2，基座用层0（层2 标签自动叫 …-ultra-beta2-pcieipc）
#   SKIP_PCIEIPC=1 bash build.sh       # 只要官方形态（层0+层1）
#   SKIP_NVAPI=1 bash build.sh         # 连包内那枚 P-State 副本的物料检查也一并跳过。
#       包 v1.2 起这个开关**不再是"不带 P-State 也能跑"的钥匙**：起容器时默认就探宿主自带那份，
#       包内副本降级成显式 opt-in（NVAPI=package），缺它只 WARN、不拦构建也不拦启动。
#   OFFICIAL=本地已有标签 MODE=layers bash build.sh    （层0 标签名）
#   BASE_IMAGE_ID=repo@sha256:<digest>                 （把 FROM 钉到 digest，别只信 tag）
#
# 复核要点：
#   1) 绝不隐式拉取：所有基础标签必须本地在位，缺就报错让人先建/先导。
#   2) 每层构建后做"终态断言"，不是"命令 rc=0 就算过"：
#      层1 断言 server.mjs 里那句超时真变成 20000 且 node --check 过；
#      层2 断言镜像里真能 import 出 PcieIpcAllReduceWorkspace（vLLM 的判据就是这一句）。
#      标签本来就已在位（没重建）时**同样跑断言**——否则第二次跑等于没验。
#   3) 回填物料先按 manifest 对拍 sha256 再进构建上下文，缺一件即停。
#   4) 失败不留半成品标签：docker build 失败本身不产标签，这里再显式确认一次。
set -u
D=$(cd "$(dirname "$0")" && pwd)
PKG=$(cd "$D/.." && pwd)

OFFICIAL=${OFFICIAL:-vllm-sm75:v0.1.7-ultra-beta2}
TO20S=${TO20S:-$OFFICIAL-to20s}
FINAL=${FINAL:-}                 # 留空=按实际层组合自动取名（SKIP_TO20S 时不该再叫 -to20s-…）
MODE=${MODE:-auto}
SKIP_TO20S=${SKIP_TO20S:-0}
SKIP_PCIEIPC=${SKIP_PCIEIPC:-0}
SKIP_NVAPI=${SKIP_NVAPI:-0}
BASE_IMAGE_ID=${BASE_IMAGE_ID:-}
NVAPI_SHA=4a199f9b259a1098ab9c01d31c67f882a2531a0fbb9c3595ad3d016c7d131d8c
NVAPI_FILE=$D/libnvidia-api.so.1
WHEEL_SHA_070=c7adf826568d61fc1b7d3aadd4cae387a35a138bfc08e2deca2f34ecfa280716
# 0.7.0.post1 的取件地址（公开 PyPI，sha256 是硬门禁，地址只是让你能自查来源）
WHEEL_URL_070=https://pypi.org/project/flashinfer-python/0.7.0.post1/

say() { echo "$*"; }
die() { echo "BUILD_FAIL $*"; exit 2; }

have() { docker image inspect "$1" >/dev/null 2>&1; }

# 真正拿去当基座的东西：给了 digest 就用 digest（tag 是可移动的，钉 digest 才知道自己建在哪之上）
base_ref() { if [ -n "$BASE_IMAGE_ID" ]; then printf '%s' "$BASE_IMAGE_ID"; else printf '%s' "$1"; fi; }

say "=== 0) 前置：物料完整性 ==="
[ -f "$D/Dockerfile.to20s-v1" ] || die "缺 docker/Dockerfile.to20s-v1"
[ -f "$D/to20s-patch-v2.sh" ] || die "缺 docker/to20s-patch-v2.sh"
[ -f "$D/Dockerfile.pcieipc-v1" ] || die "缺 docker/Dockerfile.pcieipc-v1"
if [ "$SKIP_NVAPI" = "1" ]; then
  say "  WARN SKIP_NVAPI=1：连包内那枚副本的物料检查也跳过。**这个库从来不进镜像**"
  say "       （0.1.7 的形态是宿主文件只读 bind，见 docker/NVAPI-获取说明-v1.md）。"
  say "       起容器由 run/start-here-v2.sh 决定挂哪一份：默认 NVAPI=auto 探宿主驱动自带那份，"
  say "       探不到就自动降级成\"不带库\"并把电源档同时落成 sleep——所以这个开关对启动形态**不再有直接影响**。"
  say "       推理本身（加载权重、生成）与这个库无关；它只决定面板肯不肯按 pstate 起引擎。"
else
  # 缺文件 ⇒ WARN（v1.2 的变化）。理由：默认已不依赖包内这枚副本，拿它当构建前置门等于把
  # "驱动小版本一升级就哑"的私有副本重新变成硬依赖——那正是要改掉的东西。
  # 文件在 ⇒ sha 门照跑：它是对外随包分发的二进制，被人改动过必须拦下来。
  if [ ! -f "$NVAPI_FILE" ]; then
    say "  WARN 缺 docker/libnvidia-api.so.1（包内那份 580.173.02 副本）。构建不受影响；"
    say "       起容器时默认探宿主自带那份（NVAPI=auto），探不到就落 sleep。"
    say "       要显式用包内这份就得先把文件放回原处：NVAPI=package bash run/start-here-v2.sh"
  else
    got=$(sha256sum "$NVAPI_FILE" | cut -d' ' -f1)
    [ "$got" = "$NVAPI_SHA" ] || die "包内 NVAPI 副本哈希不符 期望 $NVAPI_SHA 实为 $got（这枚二进制被改动过，别往下走）"
    say "  PASS 包内 NVAPI 副本 sha256 与清单一致（它是 opt-in 备份；默认 bind 的是宿主自带那份，且它不进镜像）"
  fi
fi

n_mat=$(awk -F'\t' '!/^#/ && $5=="verbatim"' "$D/pcieipc/manifest.txt" | wc -l | tr -dc '0-9')
[ "$n_mat" = "6" ] || die "pcieipc manifest 的 verbatim 行数=$n_mat，期望 6"
while IFS="$(printf '\t')" read -r rel _l _b sha _k; do
  p="$D/pcieipc/$rel"
  [ -f "$p" ] || die "缺回填物料 $p"
  g=$(sha256sum "$p" | cut -d' ' -f1)
  [ "$g" = "$sha" ] || die "回填物料 sha 不符 $rel 期望 $sha 实为 $g"
done < <(awk -F'\t' '!/^#/ && $5=="verbatim"' "$D/pcieipc/manifest.txt")
for b in jit_comm trace_comm comm_init; do
  [ -s "$D/pcieipc/append/$b.txt" ] || die "缺追加块 $b.txt"
done
say "  PASS 回填物料 6 件 sha 全对、3 个追加块非空"
say "  提示：这 9 件由 flashinfer_python 0.7.0.post1 wheel 解出（源 wheel sha256 $WHEEL_SHA_070）"
say "        要重新推导：python3 docker/pcieipc/extract-pcieipc.py <该 wheel> <输出目录>"

say "=== 1) 层 0：官方 0.1.7 ultra ==="
# 先证"能用 docker"，再判"镜像在不在位"。少了这一步，没有 docker 的机器会走进 else 分支，
# 报的是"找不到 0.1.7-beta2 源码树"——把人支去 clone 一个几十分钟的仓库，而真缺的是 docker；
# 源码树补上之后还要在 `docker build` 那里再撞一次。物料检查（第 0 段）不需要 docker，所以排在它后面。
command -v docker >/dev/null 2>&1 || die "PATH 上看不到 docker——这个脚本从头到尾都要它（image inspect／build／run）。处置：先装 Docker Engine，并用 docker info 确认能连上守护进程再重跑；包内物料是否完好，上面第 0 段已经核过了"
if have "$OFFICIAL"; then
  say "  已在位 $OFFICIAL -> $(docker image inspect -f '{{.Id}}' "$OFFICIAL" | cut -c8-19)，不重建"
elif [ "$MODE" = "layers" ]; then
  die "MODE=layers 但官方基座不在位：$OFFICIAL（先按官方 docker/build.sh 建，或导入）"
else
  # 层0 要你自己把上游源码树指过来（**本包不随附源码树**）。beta2 的取件与放置写法见 部署文档-v1.md 步骤 2。
  SRC=${SM75_SRC:-$PKG/code/VLLM-SM75-0.1.7-beta2}
  if [ ! -f "$SRC/docker/build.sh" ]; then
    say "  找不到 0.1.7-beta2 源码树：$SRC"
    say "  取件（上游 README 自己给的命令，分支名就是 v0.1.7-beta2）："
    say "    git clone --branch v0.1.7-beta2 --single-branch https://github.com/fishensw/VLLM-SM75.git"
    say "  放成默认路径就把它叫 code/VLLM-SM75-0.1.7-beta2；放别处就 SM75_SRC=/你的/路径 bash docker/build.sh"
    say "  已经有 vllm-sm75:v0.1.7-ultra-beta2 标签的人不需要源码树：MODE=layers bash build.sh 只做层1、层2。"
    die "找不到 0.1.7-beta2 源码树 $SRC（设 SM75_SRC 指过去，或 OFFICIAL=已有标签 MODE=layers）"
  fi
  say "  用官方入口构建：$SRC/docker/build.sh（EDITION=ultra）"
  ( cd "$SRC" && EDITION=ultra bash docker/build.sh ) || die "官方构建失败"
  have "$OFFICIAL" || die "官方构建没产出 $OFFICIAL"
fi

CUR="$OFFICIAL"
[ -n "$BASE_IMAGE_ID" ] && CUR="$(base_ref "$OFFICIAL")"

say "=== 2) 层 1：GPU 探测超时 20s ==="
if [ "$SKIP_TO20S" = "1" ]; then
  say "  SKIP_TO20S=1，跳过（面板在这台机上可能拒启：hardware.py 要 6–7 s 而它只给 3 s ⇒ 面板报「无法读取 GPU 状态」。"
  say "       判据与处置见 部署文档-v1.md 的步骤 2 与最后的「失败与处置」表第一行）"
else
  if have "$TO20S"; then
    say "  已在位 $TO20S，跳过重建，但断言照跑（不重建≠已验证）"
  else
    ctx=$(mktemp -d) || die "mktemp 失败"
    cp "$D/Dockerfile.to20s-v1" "$D/to20s-patch-v2.sh" "$ctx/" || { rm -rf "$ctx"; die "准备上下文失败"; }
    docker build -f "$ctx/Dockerfile.to20s-v1" --build-arg BASE_IMAGE="$(base_ref "$CUR")" -t "$TO20S" "$ctx" || { rm -rf "$ctx"; die "层1 构建失败"; }
    rm -rf "$ctx"
    have "$TO20S" || die "层1 没产出标签"
  fi
  docker run --rm --entrypoint sh "$TO20S" -c \
    "grep -qE 'timeout: 20000([^0-9]|\$)' /opt/sm75-workbench/console/server.mjs && node --check /opt/sm75-workbench/console/server.mjs && echo TO20S_ASSERT_OK" \
    | grep -q TO20S_ASSERT_OK || die "层1 终态断言未过（20000 没写进去或语法不过）"
  say "  PASS 层1 终态断言：server.mjs 里确为 timeout: 20000 且 node --check 过"
  CUR="$TO20S"
fi

# 层2 标签按"实际走过的层"自动命名：SKIP_TO20S=1 时基座是层0，就不该再挂 -to20s 这个名字
if [ -z "$FINAL" ] && [ "$SKIP_PCIEIPC" != "1" ]; then
  if [ "$SKIP_TO20S" = "1" ]; then FINAL="$OFFICIAL-pcieipc"; else FINAL="$TO20S-pcieipc"; fi
  say "  （层2 标签自动取名：$FINAL；要别的名字就显式 FINAL=…）"
fi

say "=== 3) 层 2：FlashInfer PCIe-IPC 回填 ==="
if [ "$SKIP_PCIEIPC" = "1" ]; then
  say "  SKIP_PCIEIPC=1，产出停在官方形态：$CUR"
  echo "BUILD_DONE IMAGE=$CUR"
  say "  这一形态下 profile 里**别**置 VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC；置了 vLLM 会打印"
  say "  this FlashInfer build does not provide PcieIpcAllReduceWorkspace 并退回原后端。"
  exit 0
fi
if have "$FINAL"; then
  say "  已在位 $FINAL，跳过重建，但断言照跑"
else
  ctx=$(mktemp -d) || die "mktemp 失败"
  cp -r "$D/pcieipc/." "$ctx/" || { rm -rf "$ctx"; die "拷上下文失败"; }
  cp "$D/Dockerfile.pcieipc-v1" "$ctx/Dockerfile" || { rm -rf "$ctx"; die "拷 Dockerfile 失败"; }
  # Dockerfile 里 COPY 的路径相对上下文根，这里把 pcieipc 内的子目录摆到位
  docker build -f "$ctx/Dockerfile" --build-arg BASE_IMAGE="$(base_ref "$CUR")" -t "$FINAL" "$ctx" || { rm -rf "$ctx"; die "层2 构建失败"; }
  rm -rf "$ctx"
  have "$FINAL" || die "层2 没产出标签"
fi
[ "$(docker image inspect -f '{{.Id}}' "$FINAL")" != "$(docker image inspect -f '{{.Id}}' "$CUR")" ] \
  || die "层2 与基座同 ID，说明补丁层没进去"
docker run --rm --entrypoint python3 "$FINAL" -c \
  "import flashinfer.comm as c,sys;sys.exit(0 if hasattr(c,'PcieIpcAllReduceWorkspace') else 1)" \
  || die "层2 终态断言未过：镜像里 import 不出 PcieIpcAllReduceWorkspace"
say "  PASS 层2 终态断言：镜像内 hasattr(flashinfer.comm,'PcieIpcAllReduceWorkspace') 为真"
CUR="$FINAL"

say "=== 4) 完成 ==="
docker image inspect "$CUR" --format 'IMAGE={{.Id}} SIZE={{.Size}}' | sed 's/^/  /'
echo "BUILD_DONE IMAGE=$CUR"
say "回填物料的源 wheel：flashinfer_python-0.7.0.post1（sha256 $WHEEL_SHA_070，取件页 $WHEEL_URL_070）"
[ "$SKIP_NVAPI" = "1" ] && say "提示：本次 SKIP_NVAPI=1 只跳过了包内那枚副本的物料检查；起容器仍按 NVAPI=auto 探宿主自带那份。"
say "下一步：编辑 run/start-here-v2.sh 顶部的路径（或用同名环境变量覆盖），然后 bash run/env-check-v1.sh && bash run/start-here-v2.sh"
