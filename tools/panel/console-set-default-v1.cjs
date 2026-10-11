#!/usr/bin/env node
// console-set-default-v1.cjs —— 把"期望默认档"恢复进面板设置（run/start-here-v2.sh 的可选步 5.9 调它）。
//
// 为什么需要它：beta2 的 server.mjs 在启动导入里有一行
//   `if (boot || !settings.defaultProfile || process.env.SM75_DEFAULT_PROFILE)
//      store.saveSettings({defaultProfile: imported.id, ...(boot ? {autoStart:true}:{})}, profiles)`
// （本地 beta2 源码树 ultra/source/console/server.mjs:1590-1591），只要容器是**带引擎参数**起的
// （docker/helpers/runtime-entrypoint.py:69-71 的 `supplied` 为真、:85 才把 SM75_LAUNCH_PROFILE 塞进 env），
// `boot` 就非空 ⇒ 每次起容器都把默认档改成它刚导入的那枚、并强开 autoStart。使用者手选的默认档会在
// 容器重启后被人不知鬼不觉地换掉。本工具只做一件事：在面板**已经答话之后**把默认档改回声明的那一枚。
//
// 语义按上游路由实现，不猜：
//   GET  /console-api/profiles      列档（先证 id 真在库里，避开 store.mjs:20 的「默认配置不存在」throw）
//   POST /console-api/settings      saveSettings 是 {...旧, ...新} 的合并语义 ⇒ 只发要改的字段，不发全量
//   GET  /console-api/settings      回读 defaultProfile 与 autoStart，与期望比对
// 两条上游护栏照抄进判据：只推 autoStart 而不带 defaultProfile 会被「自动启动需要指定默认配置」拒；
// id 不在库里会被「默认配置不存在」拒——所以这两种形态我们都在发请求之前自己判掉，不等它抛。
//
// 用法：node console-set-default-v1.cjs <profileId> [autoStart:0|1]
// 退出码（调用方必须逐码处置，不许只看"命令跑完了"）：
//   0  SET_DEFAULT_OK        已写并回读一致
//   2  ARGS_BAD              参数形状不合规（含 id 字符集不合上游删除路由）
//   3  NO_PROFILE            期望档不在库里 ⇒ 这是**声明错了**，不是"没什么要做"
//   4  API_FAIL              路由非 200／读不到设置。三种成因都归这里，输出里带返回体前 160 字以便分诊：
//                             面板没答话（连不上）、key 不对（401／403）、以及上游 POST 校验的是**合并后的全量设置**
//                             （store.mjs:20 会核 modelRoot／cacheRoot 的绝对路径与可写、source 白名单）⇒
//                             历史上那些值坏了也会返 400，**那不是"这一步没发出去"，是面板自己的设置已经不合法**
//   5  MISMATCH              写了但回读不等于期望（上游把这次 POST 吞了或改了别的字段）
// 鉴权 token 只在进程内读 /console-data/key，任何路径都不打印其值。
'use strict';
const http = require('http');
const fs = require('fs');

const BASE = process.env.CONSOLE_BASE || 'http://127.0.0.1:1615';
const ID_RE = /^[a-z0-9][a-z0-9-]{0,47}$/;      // 与上游删除路由／profile-delete.mjs:22 同一字符集

function req(method, path, body) {
  return new Promise((resolve, reject) => {
    const u = new URL(BASE + path);
    const payload = body === undefined ? null : Buffer.from(JSON.stringify(body));
    const r = http.request({
      hostname: u.hostname, port: u.port, path: u.pathname, method,
      headers: { Authorization: 'Bearer ' + fs.readFileSync('/console-data/key', 'utf8').trim(),
        'Content-Type': 'application/json' },
      timeout: 30000,
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

(async () => {
  const [, , id, wantAuto] = process.argv;
  if (!id) { console.log('ARGS_BAD 需要 profileId（可选第二参＝autoStart，只能 0 或 1）'); process.exit(2); }
  if (!ID_RE.test(id)) { console.log('ARGS_BAD profileId 形状不合规（小写字母／数字／连字符，48 字符内）: ' + id); process.exit(2); }
  let auto = null;
  if (wantAuto !== undefined) {
    if (wantAuto !== '0' && wantAuto !== '1') { console.log('ARGS_BAD autoStart 只能是 0 或 1（填了 ' + wantAuto + '）'); process.exit(2); }
    auto = wantAuto === '1';
  }

  const lp = await req('GET', '/console-api/profiles');
  if (lp.status !== 200 || !Array.isArray(lp.body)) { console.log('API_FAIL 列档没读到（status=' + lp.status + '）'); process.exit(4); }
  const hit = lp.body.find((p) => p && p.id === id);
  if (!hit) {
    // 这一条**不是**"跳过就当没事"：声明了一个不存在的档＝配置文件写错了，必须让调用方按码区分。
    console.log('NO_PROFILE 期望默认档不在库里: ' + id + '（在册 ' + lp.body.length + ' 枚；列档：node console-edit-profile-v3.cjs - list）');
    process.exit(3);
  }

  // 合并语义 ⇒ 只发要动的字段。不带 autoStart 时连这个键都不发，避免把别人设过的值顺手改了。
  const patch = { defaultProfile: id };
  if (auto !== null) patch.autoStart = auto;
  const post = await req('POST', '/console-api/settings', patch);
  if (post.status !== 200) {
    console.log('API_FAIL POST /console-api/settings 返 ' + post.status + '：' + JSON.stringify(post.body).slice(0, 160));
    process.exit(4);
  }

  const back = await req('GET', '/console-api/settings');
  if (back.status !== 200 || !back.body || typeof back.body !== 'object') {
    console.log('API_FAIL 回读设置没读到（status=' + back.status + '）'); process.exit(4);
  }
  const gotId = back.body.defaultProfile;
  const gotAuto = back.body.autoStart;
  const bad = [];
  if (gotId !== id) bad.push('defaultProfile 期望 ' + id + ' 实为 ' + JSON.stringify(gotId));
  if (auto !== null && gotAuto !== auto) bad.push('autoStart 期望 ' + auto + ' 实为 ' + JSON.stringify(gotAuto));
  if (bad.length) { console.log('MISMATCH ' + bad.join('；') + '（已 POST 但面板上的值不是这个 ⇒ 别当成功）'); process.exit(5); }
  console.log('SET_DEFAULT_OK defaultProfile=' + gotId + ' autoStart=' + JSON.stringify(gotAuto) +
    '（档名 ' + (hit.name || '') + '，本次只发 ' + Object.keys(patch).join('／') + '）');
  process.exit(0);
})().catch((e) => { console.log('API_FAIL 请求异常: ' + String(e && e.message ? e.message : e).slice(0, 160)); process.exit(4); });
