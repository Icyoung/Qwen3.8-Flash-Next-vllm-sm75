#!/bin/bash
# lint-package-v2.sh —— 分享包出厂自检（只读，不改文件；CI 式一票否决）
#
# v2 相对 v1 **只加一件事**（v1 留在包内不再被引用，语义与门禁一条都没删）：
#   C 段那条"机器专属路径"启发式（家目录前缀 ＋ 4 个字符以上）会误报**逐字随包的上游文件**——
#   上游 docker/benchlocal.sh 里给 verifier 容器挂的那枚容器内 tmpfs 目录撞的就是这条启发式，
#   而那是镜像内的目录，不是任何人的宿主路径。不能为此改上游字节（改了就不叫逐字件），
#   也不能整文件免检（那等于给泄露源开免检，v1 的 B11 刚堵过这个）。
#   v2 的做法：豁免**只对"哈希与钉死值逐字节相符"的上游逐字件生效**，且只豁免这一子项
#   （密钥形态、仓外身份清单、内网 IP 三子项照扫不误）；哈希不符 ⇒ 不但不豁免，还当场 FAIL。
#   另加 H 段独立复核这些逐字件的哈希，让"被改过的上游件"一定会红。
#
# 为什么要有：改包时最容易犯的两件事是"补丁叠加自造 bug"与"拿着过期清单当真值"。
# 这个脚本每次改包后重跑，六类检查任一失败就非零退出：
#   A 语法：所有 .sh 过 bash -n，所有 .py 过 compile
#   B 行尾/编码：文本文件必须 LF（CR=0）、UTF-8 可解码
#   C 凭据与隐私：密钥形态、用户名、内网 IP、机器专属路径 一律零命中
#   D 断链：README 与文档里引用的包内相对路径必须真实存在（含嵌图）
#   E 清单：SHA256SUMS.txt 与包内实际文件对得上（--fix 才重写）
#   F COPY 自洽：Dockerfile 里 COPY 的源，在 build.sh 组的上下文里必须真的在位
# 用法：PRIVACY_PATTERNS=../privacy-patterns.txt bash tools/lint-package-v2.sh [--fix]
set -u
D=$(cd "$(dirname "$0")/.." && pwd)
cd "$D" || exit 2
FIX=0
[ "${1:-}" = "--fix" ] && FIX=1
fail=0
# 上游逐字件：路径<TAB>钉死 sha256。新增文件必须同时改这里与 NOTICE。
UPSTREAM_VERBATIM=$'docker/benchlocal.sh\t42604393863854eded23100ba83e931125517f516a5002e0efbb24ad05902313\ndocker/VERSION\t12be7aea6ef7eab9b3c2dfe915a94f8d2ff4dc16661c349c2b12136ce0c51e00'
UV_SKIP=""          # 只对"哈希相符"的那些文件豁免路径形状那一子项
UV_BAD=0
while IFS=$'\t' read -r uv_p uv_sha; do
  [ -n "$uv_p" ] || continue
  if [ ! -f "$uv_p" ]; then echo "  FAIL 上游逐字件不在位 $uv_p"; fail=1; UV_BAD=1; continue; fi
  uv_g=$(sha256sum "$uv_p" | cut -d' ' -f1)
  if [ "$uv_g" = "$uv_sha" ]; then
    UV_SKIP="$UV_SKIP $(basename "$uv_p")"
  else
    echo "  FAIL 自称上游逐字但哈希不符 $uv_p 期望 $uv_sha 实为 $uv_g（被改过 ⇒ 不豁免，且 H 段会再报一次）"
    fail=1; UV_BAD=1
  fi
done <<EOF
$UPSTREAM_VERBATIM
EOF
bad() { echo "  FAIL $*"; fail=1; }
ok() { echo "  ok   $*"; }

# A/B 两段要跑 python 做 ast.parse 与字节级 CR/UTF-8 判定。Windows 上 `python3` 往往是
# 微软商店的空壳（跑起来返回 rc=49、什么都不输出），拿它当解释器会让 25 个文件全部"假失败"，
# 看起来像"全包都是 CRLF"。这里先探活，选一个真能跑的，探不到就直接 FAIL 而不是给假绿/假红。
PY=""
for c in ${SM75_PY_CANDIDATES:-python3 python py}; do
  command -v "$c" >/dev/null 2>&1 || continue
  "$c" -c "import sys; sys.exit(0)" >/dev/null 2>&1 || continue
  PY="$c"; break
done
[ -n "$PY" ] || { echo "LINT_FAIL 找不到能用的 python 解释器（试过 ${SM75_PY_CANDIDATES:-python3 python py}）"; exit 3; }
echo "  解释器=$PY ($("$PY" -c 'import sys;print(sys.version.split()[0])' 2>/dev/null))"

# TXT_EXT 曾用于 B 段挑文本文件，现由 find 的扩展名条件直接决定 ⇒ 删掉死变量

echo "=== A) 语法 ==="
n_sh=0
while IFS= read -r f; do
  n_sh=$((n_sh + 1))
  bash -n "$f" || bad "bash -n 失败: $f"
  # 首行 #!/bin/sh 的脚本（被 Dockerfile 的 RUN sh 消费，容器里 sh 是 dash）必须再过 sh -n：
  # bash -n 会放过 [[ ]]、数组这些在 dash 里直接 Syntax error 的 bashism。
  if [ "$(head -1 "$f")" = '#!/bin/sh' ] && command -v sh >/dev/null 2>&1; then
    sh -n "$f" || bad "sh -n 失败（该脚本按 #!/bin/sh 被消费）: $f"
  fi
done < <(find . -type f -name "*.sh" -not -path "./.git/*")
ok ".sh 文件 $n_sh 个全部过 bash -n"
n_py=0
while IFS= read -r f; do
  n_py=$((n_py + 1))
  "$PY" -c "import ast,sys;ast.parse(open(sys.argv[1],encoding='utf-8').read())" "$f" || bad "py 语法失败: $f"
done < <(find . -type f -name "*.py" -not -path "./.git/*")
ok ".py 文件 $n_py 个全部过 ast.parse"

echo "=== B) 行尾与编码 ==="
cr=0
while IFS= read -r f; do
  if "$PY" -c "
import sys
d=open(sys.argv[1],'rb').read()
if b'\r' in d: sys.exit(1)
d.decode('utf-8')
" "$f" 2>/dev/null; then :; else cr=$((cr + 1)); bad "含 CR 或非 UTF-8: $f"; fi
done < <(find . -type f \( -name "*.sh" -o -name "*.py" -o -name "*.md" -o -name "*.json" -o -name "*.txt" -o -name "*.cu" -o -name "*.cuh" -o -name "*.cjs" -o -name "*.mjs" \) -not -path "./.git/*")
[ "$cr" = "0" ] && ok "全部文本文件 LF + UTF-8"

echo "=== C) 凭据与隐私扫描 ==="
# 身份黑名单**不能写死在包内文件里**：写进来这个脚本就成了泄露源本身，而为了不让它自匹配又把它排除在
# 扫描外，等于给泄露源开免检。所以：通用形态内置，具体身份串放仓库之外的文件（PRIVACY_PATTERNS 指过去）。
# 模式里那个证书头要写成 `[E]`：原样写会让这一行自己匹配自己——检查器含被测值＝它自己就是泄露源，
# 而"为了不自匹配把它排除在扫描外"等于给泄露源开免检（本段第 ① 条堵的就是这个）。
GENERIC='(AKIA[0-9A-Z]{16}|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|[A-Za-z0-9+/]{40,}\.git|-----BEGIN CERTIFICAT[E]-----|smtp://[^ :]+:[^ @]+@|://[^ /@ ]+:[^ @]{6,}@)'
# 三条假绿通道本轮一起堵：① 不再 --exclude 自己（自匹配就改到不自匹配，不是给泄露源开免检）；
# ② 每条模式判 grep 退出码（≥2＝正则写坏，吞掉 stderr 就等于"扫过了"）；③ 扫描条数要有计数，
# 清单为空或全是注释时不许报绿。
hits=$(grep -rInE "$GENERIC" . --exclude-dir=.git --exclude=SHA256SUMS.txt 2>/dev/null | cut -d: -f1,2 | head -10)
[ -n "$hits" ] && bad "疑似密钥/令牌:$(echo "$hits" | tr '\n' ' ')" || ok "无通用密钥形态命中（本脚本自身也在扫描范围内）"
PFILE=${PRIVACY_PATTERNS:-$D/../privacy-patterns.txt}
if [ -f "$PFILE" ]; then
  n_pat=0
  while IFS= read -r pat; do
    [ -z "$pat" ] && continue
    case "$pat" in \#*) continue ;; esac
    n_pat=$((n_pat + 1))
    grep -rInE "$pat" . --exclude-dir=.git --exclude=SHA256SUMS.txt >/dev/null 2>&1
    rc=$?
    if [ "$rc" -ge 2 ]; then bad "仓外清单里这条 ERE 写坏（grep rc=$rc，正则本身有问题）：$pat"; continue; fi
    if [ "$rc" = "0" ]; then
      hits=$(grep -rInE "$pat" . --exclude-dir=.git --exclude=SHA256SUMS.txt 2>/dev/null | cut -d: -f1,2 | head -10)
      bad "命中身份/位置模式（来自仓外清单）$pat: $(echo "$hits" | tr '\n' ' ')"
    fi
  done < "$PFILE"
  if [ "$n_pat" = "0" ]; then bad "仓外清单 $PFILE 一条模式都没有 ⇒ 这一轮等于没扫"
  else ok "已按仓外 PRIVACY_PATTERNS 清单扫描 $n_pat 条模式（逐条判 grep 退出码）"; fi
else
  bad "缺仓外身份清单 $PFILE ⇒ 具体姓名/用户名/宿主路径/内网段没扫。补法：在该文件里逐行写 ERE（此文件必须在仓库之外）"
fi
# 通用"机器专属形状"内置（不含任何具体身份值）
# 反斜杠要写成字符类：写成 "盘符:\|盘符:\Users" 那种形式时，bash 递给 grep 的是 `\|`，
# 而 ERE 里 `\|` 是**字面竖线**，那条分支永不为真 ⇒ 盘符路径根本不扫（本脚本自己也在这道门的扫描范围内）。
# 路径与 IP **分两路扫**。以前是同一条件里带上 IP、再在管道尾部 `grep -vE "0\.0\.0\.0|localhost"`
# 整行放行——一行里同时写着 0.0.0.0 和真内网 IP 时整行被洗掉＝假绿。
# UV_SKIP 只影响这一子项，且只含"哈希与钉死值逐字节相符"的上游件名字；密钥/身份/IP 三子项不带这个豁免。
uv_args=""
for uv_b in $UV_SKIP; do uv_args="$uv_args --exclude=$uv_b"; done
# shellcheck disable=SC2086
hits=$(grep -rInE "(/home/|/Users/)[a-z0-9._-]{4,}|[A-Za-z]:[\\]" . --exclude-dir=.git --exclude=SHA256SUMS.txt $uv_args 2>/dev/null | cut -d: -f1,2 | head -12)
ips=$(grep -rhoE "[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}" . --exclude-dir=.git --exclude=SHA256SUMS.txt 2>/dev/null | grep -vE '^(0\.0\.0\.0|127\.0\.0\.1|255\.255\.255\.255)$' | sort -u)
[ -n "$ips" ] && bad "出现具体 IP 形状（0.0.0.0／回环／广播之外一律要核）：$(echo "$ips" | tr '\n' ' ')" || ok "无内网地址形状命中"
[ -n "$UV_SKIP" ] && ok "路径形状子项对上游逐字件放行（哈希已核）:$UV_SKIP"
[ -n "$hits" ] && bad "疑似本机专属路径/IP 形状:$(echo "$hits" | tr '
' ' ')" || ok "无本机专属路径形状命中"

echo "=== D) 文档引用断链 ==="
miss=0
while IFS= read -r ref; do
  [ -z "$ref" ] && continue
  case "$ref" in http*|\#*) continue ;; esac
  # glob 与占位写法不是链接；含非 ASCII（如省略号）的是文档示例，也不算断链
  case "$ref" in *'*'*) continue ;; esac
  case "$ref" in *'?'*) continue ;; esac
  if printf '%s' "$ref" | LC_ALL=C grep -q '[^ -~]'; then continue; fi
  p=$(printf '%s' "$ref" | sed 's|^./||')
  [ -e "$p" ] || { bad "引用不存在: $p"; miss=$((miss + 1)); }
done < <(grep -rhoE '\((\.?/?)(docker|run|tools|docs|code)/[^) ]+|`(\./)?(docker|run|tools|docs)/[^` ]+`|!\[[^]]*\]\(([^)]+)\)' \
            --include='*.md' --exclude-dir=内部 --exclude-dir=.git . 2>/dev/null \
          | sed -E 's/^!\[[^]]*\]\(//; s/^[\(`]//; s/[\)`]$//')
[ "$miss" = "0" ] && ok "文档内相对引用全部存在（含嵌图路径）"

echo "=== E) 清单一致性 ==="
if [ "$FIX" = "1" ]; then
  # Git Bash 的 sha256sum 会写成 "<hash> *<路径>"（二进制标记）。Linux 上谁重新生成都是两个空格，
  # 混着两种形状会让人以为清单被改过；这里统一成规范的两空格形态（Linux 上文本/二进制哈希同值，不影响校验）。
  # 还要排掉 ./内部/：那是过程件、被 .gitignore 挡着不进版本库。本地它存在所以 `-c` 过得去，
  # 可 clone 的人拿到的是**没有那个文件的清单** ⇒ 会看到 "FAILED open or read"，像是清单坏了。
  find . -type f -not -path "./.git/*" -not -path "./内部/*" -not -name SHA256SUMS.txt | sed 's|^\./||' | sort \
    | xargs -d '\n' sha256sum | sed -E 's/^([0-9a-f]{64}) \*/\1  /' > SHA256SUMS.txt
  ok "已重写 SHA256SUMS.txt（$(grep -c '^' SHA256SUMS.txt) 条，已排除 .gitignore 掉的过程件）"
fi
if [ -f SHA256SUMS.txt ]; then
  if sha256sum -c SHA256SUMS.txt --quiet 2>/dev/null; then ok "SHA256SUMS.txt 与包内文件一致"; else
    bad "SHA256SUMS.txt 过期或不匹配（改包后请跑 bash tools/lint-package-v2.sh --fix）"; fi
  # 覆盖门（反向数一遍）：`-c` 只照清单里已有的行去校验，所以"包内多一个文件没上清单"在它眼里是绿的，
  # 收件人跑 sha256sum -c 也只会看到"全部 OK"——那一个文件根本没被验过。删一行清单必须红。
  listed=$(awk 'NF>=2{print $2}' SHA256SUMS.txt | sed 's|^\./||' | LC_ALL=C sort)
  present=$(find . -type f -not -path "./.git/*" -not -path "./内部/*" -not -name SHA256SUMS.txt | sed 's|^\./||' | LC_ALL=C sort)
  n_present=$(printf '%s\n' "$present" | grep -c .)
  # 存在性断言排在空判据之前：find 的形状或排除项被人改坏时会数到 0，
  # 那时 comm 两个差集都空 ⇒ 这条覆盖门自己静默变成"通过"。
  [ "$n_present" -ge 20 ] || bad "E) 只数到 $n_present 个待校验文件——find 的排除项被改坏了，这条覆盖门失效"
  n_listed=$(printf '%s\n' "$listed" | grep -c .)
  [ "$n_listed" -ge 20 ] || bad "E) 清单只解出 $n_listed 条路径——清单形状或 awk 的字段假设不成立了"
  missing=$(LC_ALL=C comm -13 <(printf '%s\n' "$listed") <(printf '%s\n' "$present"))
  orphan=$(LC_ALL=C comm -23 <(printf '%s\n' "$listed") <(printf '%s\n' "$present"))
  if [ -n "${missing// /}" ]; then
    bad "E) 包内有文件没上清单（收件人的 sha256sum -c 根本不验它）: $(printf '%s' "$missing" | tr '\n' ' ')"
  elif [ -n "${orphan// /}" ]; then
    bad "E) 清单里有路径对不上包内文件（clone 后 -c 会报 FAILED open or read）: $(printf '%s' "$orphan" | tr '\n' ' ')"
  elif [ "$n_present" = "$n_listed" ]; then
    ok "清单覆盖完整（$n_present 个文件逐一对上，既没漏行也没多余行）"
  fi
  # 清单里不能出现被 .gitignore 忽略的路径（否则 clone 后 -c 必红）。
  # 这里不借 git（本地工作目录不是 git 仓，`git check-ignore` 会静默返回空＝假通过），
  # 直接拿 .gitignore 自己的条目按前缀/尾名匹配一遍。
  ghost=""
  while IFS= read -r g; do
    [ -n "$g" ] || continue
    case "$g" in *!*) continue ;; esac
    if [ "${g: -1}" = "/" ]; then
      pre="${g%/}"
      while IFS= read -r m; do
        case "$m" in "$pre"/*|*/"$pre"/*) ghost="$ghost [$m<=gitignore:$g]" ;; esac
      done < <(awk '{print $2}' SHA256SUMS.txt)
    else
      while IFS= read -r m; do
        case "$m" in $g|*/$g) ghost="$ghost [$m<=gitignore:$g]" ;; esac
      done < <(awk '{print $2}' SHA256SUMS.txt)
    fi
  done < <(grep -vE '^[[:space:]]*(#|$)' .gitignore)
  [ -z "$ghost" ] && ok "清单里没有 .gitignore 会忽略的路径" || bad "清单里写着不会进版本库的路径（clone 后校验会报缺件）:$ghost"
else bad "缺 SHA256SUMS.txt（跑 --fix 生成）"; fi

echo "=== F) build.sh 与 Dockerfile 的 COPY 自洽（B1 那一类：引用了包内不存在的名字，层根本构建不出来）==="
# build.sh 组的上下文是固定的两个形状，这里按同一个映射去解 COPY 的源路径：
#   Dockerfile.to20s-v1   -> 上下文 = docker/            （cp Dockerfile + to20s-patch-v2.sh 进去）
#   Dockerfile.pcieipc-v1 -> 上下文 = docker/pcieipc/    （cp -r pcieipc/. 到根，Dockerfile 改名放进去）
copy_ctx() { case "$1" in *Dockerfile.to20s-v1) echo docker ;; *Dockerfile.pcieipc-v1) echo docker/pcieipc ;; *) echo "" ;; esac; }
f_copy=0
# 用 find 而不是 glob：glob 空匹配时 bash 会把字面量 "docker/Dockerfile.*" 递进循环，
# 于是"一个 Dockerfile 都没检查到"会被当成"检查过了"——那正是要防的假通过。
df_list=$(find docker -maxdepth 1 -type f -name 'Dockerfile.*' | sort)
n_df=$(printf '%s
' "$df_list" | grep -c . )
for df in $df_list; do
  ctx=$(copy_ctx "$df")
  [ -n "$ctx" ] || { bad "$df 不在 build.sh 的上下文映射里（改了 build.sh 就要同步改这里）"; continue; }
  # COPY 的最后一个字段是**目标**，源是中间那些；把目标当源查会必然假失败。
  while IFS= read -r src; do
    [ -z "$src" ] && continue
    if [ "${src: -1}" = "/" ]; then
      [ -d "$ctx/$src" ] || { bad "$df 的 COPY 目录源不存在：$ctx/$src"; f_copy=$((f_copy + 1)); }
    else
      [ -e "$ctx/$src" ] || { bad "$df 的 COPY 源在上下文里不存在：$ctx/$src"; f_copy=$((f_copy + 1)); }
    fi
  done < <(grep -E '^[[:space:]]*COPY[[:space:]]' "$df" | sed -E 's/^[[:space:]]*COPY[[:space:]]+//; s/(--[a-z-]+=[^ ]+[[:space:]]+)//g; s/[[:space:]]*$//' | awk '{for(i=1;i<NF;i++) print $i}')
done
if [ "$n_df" = "0" ]; then
  bad "F) 一个 Dockerfile 都没检查到（glob 空匹配＝假通过）；docker/Dockerfile.* 改名或搬家要同步 copy_ctx 映射"
elif [ "$f_copy" != "0" ]; then bad "F) 有 $f_copy 个 COPY 源不在位"
else ok "Dockerfile 的 COPY 源在各自构建上下文里全部在位（检查 $n_df 个）"; fi

echo "=== G) 静态分析（shellcheck ＋ node --check）==="
# 为什么单独一段：A 段的 `bash -n` 只看语法，看不出"用了没定义的变量/死变量/选项位置错"。
# 0.1.7 v1.1 这轮就漏过一次：node 里引用未声明的 srcId，`node --check` 也过，一跑才炸。
# 所以这里 ① shellcheck 的 error 级当阻断、warning 级只报计数；② 每个 .cjs 过 node --check；
# ③ 工具不在位时**明写没扫**，不给假绿（与 B1 那类"glob 空匹配当检查过了"同族）。
if command -v shellcheck >/dev/null 2>&1; then
  n_sc=0; n_scerr=0
  while IFS= read -r f; do
    n_sc=$((n_sc + 1))
    out=$(shellcheck -S error -f gcc "$f" 2>&1)
    if [ -n "$out" ]; then n_scerr=$((n_scerr + 1)); bad "shellcheck error: $out"; fi
  done < <(find . -type f -name "*.sh" -not -path "./.git/*" | sort)
  n_warn=$(find . -type f -name "*.sh" -not -path "./.git/*" -exec shellcheck -S warning -f gcc {} \; 2>/dev/null | grep -c warning)
  [ "$n_scerr" = "0" ] && ok "shellcheck error 级零命中（扫 $n_sc 个 .sh；warning 级另有 $n_warn 条，属可读性，不阻断）"
  [ "$n_sc" = "0" ] && bad "G) 一个 .sh 都没扫到（find 空匹配＝假通过）"
else
  warn_tool="shellcheck 不在 PATH ⇒ 静态分析这一层没跑（对外发布前请在装了它的机器上重跑）"
  bad "$warn_tool"
fi
n_cjs=0
while IFS= read -r f; do
  n_cjs=$((n_cjs + 1))
  node --check "$f" >/dev/null 2>&1 || bad "node --check 失败: $f"
done < <(find . -type f -name "*.cjs" -not -path "./.git/*" | sort)
[ "$n_cjs" = "0" ] && bad "G) 一个 .cjs 都没扫到（tools/panel/ 空了？）" || ok ".cjs 文件 $n_cjs 个过 node --check（注意：它查不出未定义变量，那要靠跑）"

echo "=== H) 上游逐字件完整性（被改过就必须红：它不再享有 C 段的豁免，且这里独立报一次）==="
# 这一段的哈希**自己再算一遍**，不复用上面那轮的结果。上一版这里只有一句
# `if [ "$UV_BAD" = "0" ]; then ok …; fi`——坏样张进来时 UV_BAD 已经是 1，于是 H 段一条都不印，
# 标题里那句"这里独立报一次"是空的：红的只有 C 段那一行。谁改 C 段，这道"独立"的门就静默消失。
n_uv=$(printf '%s\n' "$UPSTREAM_VERBATIM" | grep -c '.')
[ "$n_uv" -ge 1 ] || bad "H) 钉住的上游逐字件清单是空的（UPSTREAM_VERBATIM 被清空＝这道门不存在了）"
while IFS=$'\t' read -r h_p h_sha; do
  [ -n "$h_p" ] || continue
  if [ ! -f "$h_p" ]; then bad "H) 上游逐字件不在位 $h_p"; continue; fi
  h_g=$(sha256sum "$h_p" | cut -d' ' -f1)
  if [ "$h_g" = "$h_sha" ]; then
    ok "H) $h_p 与钉住的哈希逐字节相符"
  else
    bad "H) 上游逐字件哈希不符 $h_p 期望 $h_sha 实为 $h_g"
  fi
done <<EOF
$UPSTREAM_VERBATIM
EOF

echo
[ "$fail" = "0" ] && { echo "LINT_PASS"; exit 0; } || { echo "LINT_FAIL 上面 FAIL 项必须清零才能对外发"; exit 3; }
