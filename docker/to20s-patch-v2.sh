#!/bin/sh
# to20s-patch-v2.sh — 断言式补丁：控制台 hardware 探测超时 3000ms → 20000ms
#
# v1 废弃原因（复核发现，未上机即拦截）：v1 用 grep -c 'timeout: 3000' 计数，
#   但 '3000' 是另一处 'timeout: 30000' 的前缀 ⇒ 计数恒为 2，"仅 1 处"断言必失败，
#   且打完补丁后的"残留检查"也永远不为 0。这是哨兵式假失败。
# v2 修法：匹配收紧为 `timeout: 3000` 后一位必须不是数字（或到行尾）；sed 仍只按行号改 314 行。
#
# 换基座（0.1.7-beta → 0.1.7-beta2）时的锚点复核（2026-10-11，本包 v1.2）：
#   在 beta2 的 console 原件（sha256 前缀 e9cd28bc）上逐条核过——
#     · 严格匹配 `timeout: 3000([^0-9]|$)`：**恰好 1 处，仍在第 314 行**（spawn hardware.py 的 options）
#     · 第 308 行那句缓存 TTL `Date.now() - hardwareAt < 3000` 不含 "timeout:" 前缀，收紧后的正则不会误计它
#     · `timeout: 30000` 那处两版都在，但行号漂了：0.1.7-beta 是 582、beta2 是 **584**
#       ⇒ 本脚本**只按出现次数比"不误伤"**（OTHER_30000_BEFORE/AFTER），不写死 582/584，所以两版都成立。
#   证据级别要如实标：以上是**本地源码副本**的复核；beta2 真实基座镜像上跑**本包这条 RUN 补丁**未做过
#   （发布机那轮是把同一处改动烘在源码树里再建镜像）。这一条登记在 CHANGELOG.md 的 v1.2"已知限制"里。
#
# 任一断言不过即非零退出，让构建直接失败，不做静默放宽。
set -eu

F=/opt/sm75-workbench/console/server.mjs
TARGET_LINE=314
# POSIX ERE：3000 后面不能紧跟数字，否则会把 30000 误计
PAT_3000='timeout: 3000([^0-9]|$)'
PAT_20000='timeout: 20000([^0-9]|$)'

test -f "$F" || { echo "ASSERT_FAIL 文件不存在 $F"; exit 2; }
echo "BEFORE_SHA=$(sha256sum "$F" | cut -d' ' -f1)"

hits=$(grep -cE "$PAT_3000" "$F" || true)
if [ "$hits" != "1" ]; then
  echo "ASSERT_FAIL 期望严格匹配 timeout: 3000 仅 1 处，实为 $hits（行号可能已变，别盲改）"
  grep -nE "$PAT_3000" "$F" | head -5
  exit 3
fi
# 记录其它超时值（timeout: 30000，0.1.7-beta 在第 582 行、beta2 漂到第 584 行）出现次数，改完后必须一模一样
other_before=$(grep -c 'timeout: 30000' "$F" || true)
echo "OTHER_30000_BEFORE=$other_before"

line=$(sed -n "${TARGET_LINE}p" "$F")
case "$line" in
  *"timeout: 3000"*) : ;;
  *) echo "ASSERT_FAIL 第 ${TARGET_LINE} 行不是目标行: $line"; exit 4 ;;
esac

sed -i "${TARGET_LINE}s/timeout: 3000/timeout: 20000/" "$F"

left=$(grep -cE "$PAT_3000" "$F" || true)
if [ "$left" != "0" ]; then echo "ASSERT_FAIL 仍残留严格匹配的 timeout: 3000 共 $left 处"; exit 5; fi
grep -qE "$PAT_20000" "$F" || { echo 'ASSERT_FAIL 未写入 timeout: 20000'; exit 6; }
echo "AFTER_LINE=$(sed -n "${TARGET_LINE}p" "$F" | tr -d ' ')"

# 其它超时值必须一处都没被误伤（不写死 582 行号，改数出现次数）
other_after=$(grep -c 'timeout: 30000' "$F" || true)
echo "OTHER_30000_AFTER=$other_after"
if [ "$other_after" != "$other_before" ]; then
  echo "ASSERT_FAIL timeout: 30000 出现次数被改动：$other_before -> $other_after"
  exit 7
fi

# 语法必须仍可加载
node --check "$F"
echo "AFTER_SHA=$(sha256sum "$F" | cut -d' ' -f1)"
echo "TO20S_PATCH_OK"
