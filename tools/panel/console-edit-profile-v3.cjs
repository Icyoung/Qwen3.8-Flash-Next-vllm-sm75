// console-edit-profile-v3.cjs — 带护栏地改/建面板 profile
//
// v3 相对 v2 **只改一处**：clone 的新档 id 字符集收紧成 /^[a-z0-9][a-z0-9-]{0,47}$/。
//   为什么要改：上游 0.1.7-beta2 的删除路由匹配 /^\/api\/profiles\/([a-z0-9-]+)\/(.+)$/，
//   而 profile-delete.mjs:22 自己校验的也是同一套字符集——v2 放行的 `.` 与 `_` 会建出**建得出来、删不掉**的孤档
//   （走 POST 是 upsert，只能靠再 POST 一份同 id 覆盖，改不了 id 本身）。
//   v2 因此作废（文件保留在 tools/panel/ 里不再被任何脚本引用）；除这一处校验外，v2 的十二种模式与全部护栏逐字未动。
//
// v1 的三种模式（setenv / rmenv / setarg）保留原语义，v2 新增：
//   addarg  <flag> <value>   追加一个当前档里不存在的参数（v1 的 setarg 明确拒绝新增，
//                            而 --kv-transfer-config 这类"多人档才需要"的项本来就不在模板里）
//   addswitch <flag>         追加一个开关型参数（无值）
//   delarg  <flag>           删除一个"带值"的参数（flag＋值一起删）
//   delswitch <flag>         删除一个开关型参数（后面紧跟另一个 flag 的那种）
//   setfield <path> <value>  改顶层/嵌套字段（power.idleSeconds 这类既不是 env 也不是 argv）
//   setmodel <绝对路径>       改 args[0]（位置参数，setarg 够不着）
//   list                     只读：所有档的 id／名字／argv 长度
//   dump    <profileId>      只读：KEY\tVALUE 扁平行，供 shell 消费
//   clone   <srcId> <newId> <newName> <jsonFile>   从包内 JSON 建一档（面板 POST 是 upsert）
//
// 护栏（与 v1 同源，一条不放宽）：
//   ① 引擎在跑 ⇒ 拒绝；② 每次写后回读整档逐字段比对，除本次声明的那一处外任何变化 ⇒ POST 原档回滚并非零退出；
//   ③ 密钥只进请求头，值一律脱敏；④ 不猜字段类型：setfield 走白名单，addarg/delarg 先判开关还是带值。
// 退出码：2 参数不合规 3 引擎在跑 4 找不到档 5 flag 已存在/不存在 6 形状判不了 7 POST 失败
//         8 回读不干净（已尝试回滚） 9 档丢了 10 目标 id 已在位（clone） 11 白名单外字段
const http = require('http');
const fs = require('fs');

const BASE = process.env.CONSOLE_BASE || 'http://127.0.0.1:1615';
const SECRET = /key|token|secret|password|passwd|credential|authorization|api[-_]?key/i;
const SECRET_VALUE = /^[A-Za-z0-9_+/-]{40,}={0,2}$|^[0-9a-fA-F]{32,}$/;
const SECRET_ARG = /^-{1,2}(api[-_]?key|token|secret|password|credential)$/i;
// setfield 白名单：路径 ⇒ 校验器。不在表里的路径一律拒（"能改一切"的通道不是配置器，是后门）
const FIELD_OK = {
  'power.idleSeconds': (v) => /^\d+$/.test(v) ? Number(v) : null,
  'power.mode': (v) => (v === 'pstate' || v === 'sleep' ? v : null),
  'power.util': (v) => /^\d+$/.test(v) ? Number(v) : null,
  name: (v) => (v.length > 0 && v.length <= 120 ? v : null),
  defaultModel: (v) => (v.startsWith('/') ? v : null),
  format: (v) => (/^[a-z0-9_-]+$/.test(v) ? v : null),
};
function looksSecret(v) { return typeof v === 'string' && SECRET_VALUE.test(v); }
// 判"这个 token 是不是密钥类 flag"，两种形态都要认：`--api-key X` 与 `--api-key=X`。
// 只认前一种会漏：等号形态里密钥就在 token 自己身上，脱敏形同虚设。
// 注意这里**只用锚定的 SECRET_ARG**：flag 名上跑宽松的 /token|key|secret/ 子串匹配，
// 会把 `--max-num-batched-tokens` 这种正常参数也遮成 ***，调用方读不到当前值 ⇒ 每次都会多发一条写命令。
// env 的键名（OPENAI_API_KEY 这类）才用宽松匹配——那里宁可多遮。
function secretFlagToken(tok) {
  const t = String(tok).replace(/^-+/, '').split('=')[0];
  if (!t) return false;
  return SECRET_ARG.test('--' + t);
}
function showArg(args, i) {
  const raw = cellOf((args || [])[i]);
  if (raw.includes('=') && secretFlagToken(raw)) return '***';
  if (secretFlagToken((args || [])[i - 1]) || looksSecret(raw)) return '***';
  return raw;
}
function req(method, path, body) {
  return new Promise((resolve, reject) => {
    const u = new URL(BASE + path);
    const payload = body === undefined ? null : Buffer.from(JSON.stringify(body));
    const r = http.request({
      hostname: u.hostname, port: u.port, path: u.pathname, method,
      headers: { Authorization: 'Bearer ' + fs.readFileSync('/console-data/key', 'utf8').trim() },
      timeout: 60000,
    }, (res) => {
      let d = '';
      res.on('data', (c) => (d += c));
      res.on('end', () => { let p = d; try { p = JSON.parse(d); } catch (e) {} resolve({ status: res.statusCode, body: p }); });
    });
    r.on('timeout', () => r.destroy(new Error('timeout')));
    r.on('error', reject);
    if (payload) r.write(payload);
    r.end();
  });
}
const list = async () => ((await req('GET', '/console-api/profiles')).body || []);
const getOne = async (id) => (await list()).find((x) => x.id === id);
function envDiff(a, b) {
  const out = [];
  for (const k of new Set([...Object.keys(a || {}), ...Object.keys(b || {})])) {
    if (String((a || {})[k]) !== String((b || {})[k])) out.push('env.' + k);
  }
  return out.sort();
}
function argDiff(a, b) {
  a = a || []; b = b || [];
  if (a.length !== b.length) return ['args_LENGTH(' + a.length + '->' + b.length + ')'];
  const out = [];
  for (let i = 0; i < a.length; i++) if (cellOf(a[i]) !== cellOf(b[i])) out.push('args[' + i + ']');
  return out;
}
function otherDiff(a, b) {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  const out = [];
  for (const k of keys) {
    if (k === 'env' || k === 'args') continue;
    if (JSON.stringify(a[k]) !== JSON.stringify(b[k])) out.push(k);
  }
  return out.sort();
}
function shown(p) {
  const e = {};
  const a = ((p || {}).args || []).map((v, i, arr) => showArg(arr, i));
  for (const [k, v] of Object.entries((p || {}).env || {})) e[k] = SECRET.test(k) || looksSecret(v) ? '***' : v;
  return { args: a, env: e };
}
// argv 里既有字符串也有 JSON 对象（--compilation-config 这类）。
// 一律 String() 会把对象印成 [object Object]、把内容比对变成"永远相等"⇒ 面板改了内容也判不出。
function cellOf(v) { return typeof v === 'object' && v !== null ? JSON.stringify(v) : String(v); }
function flat(p) {
  const lines = ['PROFILE_ID\t' + p.id, 'ARGV_LEN\t' + (p.args || []).length,
    'ARGV0\t' + cellOf((p.args || [])[0] ?? 'ABSENT'),
    'FIELD_power_idleSeconds\t' + ((p.power || {}).idleSeconds ?? 'ABSENT'),
    'FIELD_power_mode\t' + ((p.power || {}).mode ?? 'ABSENT'),
    'FIELD_name\t' + (p.name ?? 'ABSENT'), 'FIELD_format\t' + (p.format ?? 'ABSENT'),
    'FIELD_defaultModel\t' + (p.defaultModel ?? 'ABSENT')];
  for (let i = 0; i < (p.args || []).length; i++) {
    const ftok = String(p.args[i]);
    if (!ftok.startsWith('--')) continue;
    // 等号形态的 flag：值就在 token 里，那一列本身也要遮
    const fshow = ftok.includes('=')
      ? (secretFlagToken(ftok) ? ftok.split('=')[0] + '=<已遮>' : ftok) : ftok;
    const nxt = p.args[i + 1];
    const isSwitch = nxt === undefined || String(nxt).startsWith('--');
    lines.push('FLAG\t' + fshow + '\t' + (isSwitch ? '(SWITCH)' : showArg(p.args, i + 1)));
  }
  for (const [k, v] of Object.entries(p.env || {})) {
    lines.push('ENV\t' + k + '\t' + (SECRET.test(k) || looksSecret(String(v)) ? '***' : cellOf(v)));
  }
  return lines.join('\n') + '\n';
}
// 位置参数（args[0]＝模型目录）不是 flag，单独给一条通道，别拿 setarg 去凑
function isFlag(a) { return String(a).startsWith('--'); }
function flagIndex(args, flag) { return (args || []).findIndex((a) => String(a) === flag); }

(async () => {
  const [, , id, mode, a1, a2, a3, a4] = process.argv;
  if (!id || !mode) { console.log('ARGS_BAD 需要 profileId 与模式'); process.exit(2); }

  if (mode === 'list') {
    for (const p of await list()) console.log(p.id + '\t' + (p.name || '') + '\t' + (p.args || []).length);
    console.log('LIST_OK');
    return;
  }

  if (mode === 'dump') {
    const p = await getOne(id);
    if (!p) { console.log('ABORT 找不到 profile ' + id); process.exit(4); }
    process.stdout.write(flat(p));
    console.log('DUMP_OK');
    return;
  }

  if (mode === 'clone') {
    // 位置沿用全局解构：id＝母本（没有母本传 "-"），a1/a2/a3＝新 id／新名／JSON 路径
    const srcId = id, newId = a1, newName = a2, file = a3;
    // clone 是写操作，护栏①对它同样成立（原来它排在状态检查之前 return＝引擎在跑也能建档）
    const st0 = await req('GET', '/console-api/profiles/' + (srcId === '-' ? newId : srcId) + '/status');
    if (st0.body && st0.body.running) { console.log('ABORT 引擎在跑，先停'); process.exit(3); }
    if (!newId || !newName || !file) { console.log('ARGS_BAD clone 需要 <newId> <newName> <jsonFile>'); process.exit(2); }
    // 字符集对齐上游删除路由（server.mjs:1353 的正则与 profile-delete.mjs:22 的校验都是 [a-z0-9-]、48 字符内）：
    // 带 `.` 或 `_` 的 id 建出来之后走不到那条路由＝永远删不掉的孤档。
    if (!/^[a-z0-9][a-z0-9-]{0,47}$/.test(newId)) { console.log('ARGS_BAD newId 形状不合规（只能小写字母、数字与连字符，48 字符内；带 . 或 _ 的档走不到上游删除路由）: ' + newId); process.exit(2); }
    if (!file.startsWith('/console-data/')) { console.log('ARGS_BAD jsonFile 必须在 /console-data/ 下（容器里读得到的 bind）'); process.exit(2); }
    // startsWith 不拦 `..`：/console-data/../../etc/passwd 照样过前缀门 ⇒ 逐段查
    if (file.split('/').includes('..')) { console.log('ARGS_BAD jsonFile 不允许含 .. 段'); process.exit(2); }
    if (!fs.existsSync(file)) { console.log('ARGS_BAD jsonFile 不在位: ' + file); process.exit(2); }
    if (await getOne(newId)) { console.log('ABORT 目标 id 已在位，面板 POST 是 upsert＝会覆盖别人那一档：' + newId); process.exit(10); }
    let want;
    try { want = JSON.parse(fs.readFileSync(file, 'utf8')); }
    // Node 的 JSON.parse 错误消息会带文件开头若干字节——读错文件时那就是内容泄露，一律吞掉
    catch (e) { console.log('ARGS_BAD jsonFile 不是合法 JSON：' + file); process.exit(2); }
    if (!Array.isArray(want.args) || !want.args.length) { console.log('ARGS_BAD jsonFile 里没有 args'); process.exit(2); }
    // 包内 JSON 的 args[0] 是模板占位符 ${MODEL}（官方模板就这么存）。原样建进去 ⇒ 引擎拿一个
    // 叫 "${MODEL}" 的模型目录去加载，表现是"档建好了、起不来"。所以占位符必须在这里换掉。
    if (/^\$\{[A-Z_]+\}$/.test(String(want.args[0]))) {
      if (!a4 || !a4.startsWith('/')) {
        console.log('ARGS_BAD 包内档的 args[0] 是占位符 ' + want.args[0] + '，clone 必须再给第 4 个参数＝模型绝对路径');
        process.exit(2);
      }
      want = JSON.parse(JSON.stringify(want));
      want.args[0] = a4;
      if (typeof want.defaultModel === 'string' && /^\$\{[A-Z_]+\}$/.test(want.defaultModel)) want.defaultModel = a4;
      console.log('MODEL_SET=' + a4);
    }
    let src = null;
    if (srcId !== '-') {
      src = await getOne(srcId);
      if (!src) { console.log('ABORT 找不到母本 profile ' + srcId); process.exit(4); }
    }
    const base = src ? JSON.parse(JSON.stringify(src)) : JSON.parse(JSON.stringify(want));
    base.id = newId;
    base.name = newName;
    // 建档只允许动这三处（id/name 由命令给，args/env 由 JSON 给）；JSON 里其它顶层字段整份带过来
    for (const k of ['args', 'env']) if (want[k] !== undefined) base[k] = JSON.parse(JSON.stringify(want[k]));
    const post = await req('POST', '/console-api/profiles', base);
    if (post.status !== 200) { console.log('POST_FAIL status=' + post.status); process.exit(7); }
    const after = await getOne(newId);
    if (!after) { console.log('LOST_PROFILE 新档回读不到'); process.exit(9); }
    const d = [...envDiff(base.env, after.env), ...argDiff(base.args, after.args), ...otherDiff(base, after)]
      .filter((x) => x !== 'id' && x !== 'name');
    if (d.length) { console.log('UNEXPECTED=' + d.join(',')); process.exit(8); }
    console.log('CLONE_OK ' + newId + ' args=' + (after.args || []).length);
    return;
  }

  const st = await req('GET', '/console-api/profiles/' + id + '/status');
  if (st.body && st.body.running) { console.log('ABORT 引擎在跑，先停'); process.exit(3); }
  const before = await getOne(id);
  if (!before) { console.log('ABORT 找不到 profile ' + id); process.exit(4); }
  const snapshot = JSON.parse(JSON.stringify(before));
  const next = JSON.parse(JSON.stringify(before));
  next.env = Object.assign({}, next.env || {});
  let expected = [];
  // expectArgs：addarg/delarg/delswitch 会改长度，逐位比对不适用，改成"整段应等于我算出来的数组"
  let expectArgs = null;

  if (mode === 'setenv' || mode === 'rmenv') {
    if (!/^[A-Z][A-Z0-9_]{2,}$/.test(a1 || '')) { console.log('ARGS_BAD ENV KEY 不合规: ' + a1); process.exit(2); }
    if (mode === 'setenv') {
      if (a2 === undefined) { console.log('ARGS_BAD setenv 需要值'); process.exit(2); }
      next.env[a1] = a2;
    } else {
      if (!Object.prototype.hasOwnProperty.call(next.env, a1)) { console.log('NOOP rmenv 该键本来就没有: ' + a1); process.exit(0); }
      delete next.env[a1];
    }
    expected = ['env.' + a1];
  } else if (mode === 'setarg') {
    if (!/^--[a-z0-9-]+$/.test(a1 || '')) { console.log('ARGS_BAD flag 不合规: ' + a1); process.exit(2); }
    if (a2 === undefined) { console.log('ARGS_BAD setarg 需要新值'); process.exit(2); }
    const i = flagIndex(next.args, a1);
    if (i < 0) { console.log('ABORT setarg 只允许改已有参数，档里没有 ' + a1 + '（新增请走 addarg）'); process.exit(5); }
    if (i + 1 >= next.args.length || isFlag(next.args[i + 1])) { console.log('ABORT ' + a1 + ' 看起来是开关，不敢当带值参数改'); process.exit(6); }
    console.log('OLD_ARG[' + (i + 1) + ']=' + showArg(next.args, i + 1));
    next.args[i + 1] = a2;
    expected = ['args[' + (i + 1) + ']'];
  } else if (mode === 'setmodel') {
    // args[0] 是位置参数（模型目录），不是 flag，setarg 找不到它，所以单开一条通道
    if (!/^\//.test(a1 || '') || /[;&|`$"'\\]/.test(a1 || '')) { console.log('ARGS_BAD setmodel 要绝对路径且不含 shell 特殊字符: ' + a1); process.exit(2); }
    if (!next.args || !next.args.length) { console.log('ABORT 档里没有 args[0]，不敢凭空造模型位置'); process.exit(6); }
    console.log('OLD_MODEL=' + showArg(next.args, 0));
    next.args[0] = a1;
    expected = ['args[0]'];
  } else if (mode === 'addarg') {
    if (!/^--[a-z0-9-]+$/.test(a1 || '')) { console.log('ARGS_BAD flag 不合规: ' + a1); process.exit(2); }
    if (a2 === undefined) { console.log('ARGS_BAD addarg 需要值（开关请走 addswitch）'); process.exit(2); }
    if (flagIndex(next.args, a1) >= 0) { console.log('ABORT ' + a1 + ' 已存在，改值请走 setarg'); process.exit(5); }
    next.args = (next.args || []).concat([a1, a2]);
    expectArgs = next.args.slice();
    expected = ['args_APPEND(' + a1 + ')'];
  } else if (mode === 'addswitch') {
    if (!/^--[a-z0-9-]+$/.test(a1 || '')) { console.log('ARGS_BAD flag 不合规: ' + a1); process.exit(2); }
    if (flagIndex(next.args, a1) >= 0) { console.log('ABORT ' + a1 + ' 已存在'); process.exit(5); }
    next.args = (next.args || []).concat([a1]);
    expectArgs = next.args.slice();
    expected = ['args_APPEND(' + a1 + ')'];
  } else if (mode === 'delarg' || mode === 'delswitch') {
    if (!/^--[a-z0-9-]+$/.test(a1 || '')) { console.log('ARGS_BAD flag 不合规: ' + a1); process.exit(2); }
    const i = flagIndex(next.args, a1);
    if (i < 0) { console.log('NOOP delarg 档里本来就没有 ' + a1); process.exit(0); }
    const nxt = next.args[i + 1];
    const hasValue = nxt !== undefined && !isFlag(nxt);
    if (mode === 'delarg' && !hasValue) { console.log('ABORT ' + a1 + ' 后面没有值（开关请走 delswitch）'); process.exit(6); }
    if (mode === 'delswitch' && hasValue) { console.log('ABORT ' + a1 + ' 带值（请走 delarg，别把它的值留给下一项）'); process.exit(6); }
    console.log('OLD_ARG[' + (i + 1) + ']=' + (hasValue ? showArg(next.args, i + 1) : '(SWITCH)'));
    next.args.splice(i, hasValue ? 2 : 1);
    expectArgs = next.args.slice();
    expected = ['args_REMOVE(' + a1 + ')'];
  } else if (mode === 'setfield') {
    const chk = FIELD_OK[a1 || ''];
    if (!chk) { console.log('FIELD_DENIED 白名单外的字段: ' + a1); process.exit(11); }
    if (a2 === undefined) { console.log('ARGS_BAD setfield 需要值'); process.exit(2); }
    const v = chk(a2);
    if (v === null) { console.log('ARGS_BAD ' + a1 + ' 值不合规: ' + a2); process.exit(2); }
    const parts = a1.split('.');
    const old = parts.reduce((o, k) => (o == null ? o : o[k]), next);
    console.log('OLD_FIELD=' + a1 + '=' + (old === undefined ? 'ABSENT' : old));
    let root = next;
    for (let i = 0; i < parts.length - 1; i++) root[parts[i]] = Object.assign({}, root[parts[i]] || {}), root = root[parts[i]];
    root[parts[parts.length - 1]] = v;
    expected = [parts[0]];
  } else { console.log('UNKNOWN_MODE ' + mode); process.exit(2); }

  // 幂等门：算出来的 next 与当前档逐字段一致 ⇒ 什么都不发。
  // 用 cellOf 逐位比（对象与其 JSON 文本视作同一个值），否则"把对象换成同样内容的字符串"
  // 会既过不了这道门、又过不了后面的回读断言，来回自相矛盾。
  const sa = snapshot.args || [], na = next.args || [];
  const argsSame = sa.length === na.length && sa.every((v, i) => cellOf(v) === cellOf(na[i]));
  if (argsSame
    && JSON.stringify(next.env) === JSON.stringify(snapshot.env)
    && otherDiff(snapshot, next).length === 0) {
    console.log('NOOP ' + mode + ' 目标值与当前一致，未 POST');
    process.exit(0);
  }

  const post = await req('POST', '/console-api/profiles', next);
  if (post.status !== 200) {
    const pb = post.body && typeof post.body === 'object' && !Array.isArray(post.body)
      ? { args: post.body.args, env: post.body.env || post.body } : { args: [], env: {} };
    console.log('POST_FAIL status=' + post.status + ' body=' + JSON.stringify(shown(pb)).slice(0, 400));
    process.exit(7);
  }
  const after = await getOne(id);
  if (!after) { console.log('LOST_PROFILE 回读不到，尝试回滚'); after_missing(); }

  let got;
  if (expectArgs) {
    const same = JSON.stringify(after.args.map(cellOf)) === JSON.stringify(expectArgs.map(cellOf));
    // 增删类改的是长度，逐位比对不适用 ⇒ 用"整段等于我算出来的数组"当定义级断言。
    // 成立时要把声明的那个 args_* 记进 got，否则下面 missing 判定会把一次成功的改动判成失败（假红）。
    got = [...envDiff(snapshot.env, after.env),
      ...(same ? expected.slice(0, 1)
        : ['args_MISMATCH@' + (after.args || []).findIndex((v, i) => cellOf(v) !== cellOf(expectArgs[i]))]),
      ...otherDiff(snapshot, after)];
  } else {
    got = [...envDiff(snapshot.env, after.env), ...argDiff(snapshot.args, after.args), ...otherDiff(snapshot, after)];
  }
  got = got.sort();
  const unexpected = got.filter((d) => !expected.includes(d));
  const missing = expected.filter((e) => !got.includes(e));
  console.log('EXPECTED=' + expected.join(',') + '  GOT=' + (got.join(',') || 'none'));
  if (expectArgs) console.log('NEW_ARGV_LEN=' + (after.args || []).length);
  if (mode === 'setarg' || mode === 'delarg') console.log('NOW_FLAG=' + a1);
  if (mode === 'setenv') console.log('NOW_ENV=' + (SECRET.test(a1) || looksSecret(String(after.env[a1])) ? '***' : after.env[a1]));
  if (mode === 'setfield') console.log('NOW_FIELD=' + a1 + '=' + a1.split('.').reduce((o, k) => (o == null ? o : o[k]), after));
  // 终态断言：删除类必须"真不在"，新增类必须"真在"——只比 diff 不够，面板可能把整段吞了
  if ((mode === 'delarg' || mode === 'delswitch') && flagIndex(after.args, a1) >= 0) {
    console.log('ASSERT_FAIL_ABSENT ' + a1 + ' 删完还在位'); process.exit(8);
  }
  if ((mode === 'addarg' || mode === 'addswitch') && flagIndex(after.args, a1) < 0) {
    console.log('ASSERT_FAIL_PRESENT ' + a1 + ' 声称加了但档里找不到'); process.exit(8);
  }

  if (unexpected.length || missing.length) {
    console.log('UNEXPECTED=' + unexpected.join(',') + ' MISSING=' + missing.join(','));
    console.log('BEFORE=' + JSON.stringify(shown(snapshot)).slice(0, 900));
    console.log('AFTER=' + JSON.stringify(shown(after)).slice(0, 900));
    const rb = await req('POST', '/console-api/profiles', snapshot);
    const back = await getOne(id);
    const resid = [...envDiff(snapshot.env, back.env), ...argDiff(snapshot.args, back.args), ...otherDiff(snapshot, back)];
    console.log('ROLLBACK_POST=' + rb.status + ' RESIDUAL=' + (resid.join(',') || 'none'));
    process.exit(8);
  }
  console.log('EDIT_OK 仅声明的那一处变了，其余逐字段一致');
})().catch((e) => { console.log('DRIVER_ERROR ' + e.message); process.exit(1); });

function after_missing() { console.log('ABORT 档位丢了'); process.exit(9); }
