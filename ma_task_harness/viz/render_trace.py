"""Render `harness_trace.json` as a unified, chronological multi-agent dialogue HTML viewer.

Presents the entire Stage 1 -> Stage 2 (G^(n) <-> E^(n) + SAPIEN/PhysX) -> Stage 3
lifecycle as ONE clear chat stream so any reader can follow every interaction top-to-bottom:
  - User bubbles (right-aligned): task idea & clarification answers
  - Specifier bubbles (purple): Why/When/Who clarifying questions & validated TaskSpec.yaml
  - Generator G^(n) bubbles (blue): scene/expert summary, collapsible unified code diffs,
    and highlighted G->E Counter-Audit bubbles when emitted
  - Evaluator E^(n) bubbles (terracotta): deterministic detector summary & code diffs
  - SAPIEN + PhysX Simulator cards (centered): real GPU-rendered RGB playback (play/pause +
    scrubber + thumbnail filmstrip), physical gate badges, (S_k, S_collab, SR) scores,
    and concrete failure diagnostics
  - Final Stage 3 Certification & Frozen Package card at the end of the thread
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


_CHAT_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>MA-Task Harness 对话轨迹 — __TITLE__</title>
<style>
  :root {
    --bg: #eef1f7;
    --surface: #ffffff;
    --ink: #181c27;
    --muted: #5b6478;
    --border: #dce2ee;
    --user-bg: #2563eb;
    --user-fg: #ffffff;
    --spec-bg: #f5f3ff;
    --spec-bd: #c4b5fd;
    --spec-fg: #5b21b6;
    --gen-bg: #eff6ff;
    --gen-bd: #93c5fd;
    --gen-fg: #1d4ed8;
    --eval-bg: #fff7ed;
    --eval-bd: #fdba74;
    --eval-fg: #c2410c;
    --sim-bg: #f8fafc;
    --sim-bd: #cbd5e1;
    --ok: #15803d;
    --ok-bg: #dcfce7;
    --fail: #b91c1c;
    --fail-bg: #fee2e2;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "PingFang SC", "Segoe UI", Roboto, sans-serif;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: var(--sans); background: var(--bg); color: var(--ink); line-height: 1.55; }

  /* Sticky Top Header + Jump Bar */
  .topbar { position: sticky; top: 0; z-index: 30; background: #111827; color: #f8fafc; padding: 14px 24px 10px; border-bottom: 2px solid #2563eb; box-shadow: 0 2px 10px rgba(0,0,0,0.16); }
  .topbar-inner { max-width: 980px; margin: 0 auto; }
  .top-row { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px; }
  .top-row h1 { margin: 0; font-size: 18px; font-weight: 700; }
  .pill-row { display: flex; flex-wrap: wrap; gap: 6px; }
  .pill { font-size: 11.5px; font-weight: 600; padding: 3px 9px; border-radius: 999px; background: #1f2937; border: 1px solid #374151; color: #e5e7eb; }
  .pill.ok { background: #14532d; border-color: #22c55e; color: #dcfce7; }
  .pill.fail { background: #7f1d1d; border-color: #ef4444; color: #fee2e2; }
  .pill.blue { background: #1e3a8a; border-color: #60a5fa; color: #dbeafe; }

  .jumpbar { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; padding-top: 8px; border-top: 1px solid #1f2937; }
  .jumpbar a { color: #cbd5e1; text-decoration: none; font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 6px; background: #1f2937; }
  .jumpbar a:hover { background: #374151; color: #fff; }

  /* Unified Chat Stream Container */
  .chat-stream { max-width: 980px; margin: 22px auto 80px; padding: 0 18px; display: flex; flex-direction: column; gap: 14px; }

  /* Stage Divider */
  .divider { display: flex; align-items: center; gap: 12px; margin: 18px 0 4px; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }
  .divider::before, .divider::after { content: ""; flex: 1; height: 1px; background: var(--border); }

  /* Chat Rows & Bubbles */
  .row { display: flex; align-items: flex-start; gap: 11px; }
  .row.right { flex-direction: row-reverse; }
  .avatar { width: 38px; height: 38px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 800; flex-shrink: 0; border: 2px solid transparent; }
  .av-user { background: var(--user-bg); color: #fff; }
  .av-spec { background: var(--spec-bg); color: var(--spec-fg); border-color: var(--spec-bd); }
  .av-gen  { background: var(--gen-bg);  color: var(--gen-fg);  border-color: var(--gen-bd); }
  .av-eval { background: var(--eval-bg); color: var(--eval-fg); border-color: var(--eval-bd); }
  .av-sim  { background: #0f172a; color: #38bdf8; border-color: #38bdf8; }

  .bubble { max-width: 84%; border-radius: 14px; padding: 13px 16px; border: 1px solid var(--border); background: var(--surface); box-shadow: 0 1px 3px rgba(15,23,42,0.05); }
  .bubble.user { background: var(--user-bg); color: var(--user-fg); border-color: #1d4ed8; border-top-right-radius: 4px; }
  .bubble.spec { background: var(--spec-bg); border-color: var(--spec-bd); border-top-left-radius: 4px; }
  .bubble.gen  { background: var(--gen-bg);  border-color: var(--gen-bd);  border-top-left-radius: 4px; }
  .bubble.eval { background: var(--eval-bg); border-color: var(--eval-bd); border-top-left-radius: 4px; }
  .bubble.counter { background: #fff1f2; border: 2px solid #fb7185; border-top-left-radius: 4px; }

  .b-head { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 7px; font-size: 12px; font-weight: 700; }
  .bubble.user .b-head { color: #dbeafe; }
  .b-head .role { display: inline-flex; align-items: center; gap: 6px; }
  .b-head .meta { font-weight: 500; opacity: 0.78; font-family: var(--mono); font-size: 11px; }
  .b-body { font-size: 14px; white-space: pre-wrap; word-break: break-word; }
  .b-body ol { margin: 6px 0 0; padding-left: 20px; }
  .b-body li { margin-bottom: 4px; }

  /* Full-width Simulator / Gate / Probe Cards in the chat stream */
  .sim-card { background: var(--surface); border: 1.5px solid var(--sim-bd); border-radius: 14px; padding: 16px 18px; box-shadow: 0 2px 6px rgba(15,23,42,0.06); }
  .sim-card.pass { border-color: #22c55e; }
  .sim-card.fail { border-color: #f87171; }
  .sim-head { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 12px; }
  .sim-title { font-size: 15px; font-weight: 700; display: flex; align-items: center; gap: 8px; }

  .badges { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
  .badge { font-size: 12px; font-weight: 600; padding: 3px 9px; border-radius: 6px; }
  .badge.ok { background: var(--ok-bg); color: var(--ok); }
  .badge.fail { background: var(--fail-bg); color: var(--fail); }
  .badge.info { background: #e0f2fe; color: #0369a1; }
  .badge.warn { background: #fef3c7; color: #92400e; }

  /* Real SAPIEN RGB Video/Keyframe Player */
  .player-box { background: #0b1220; border-radius: 10px; padding: 12px; color: #e2e8f0; margin-bottom: 12px; }
  .player-views { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; }
  .view-pane { text-align: center; }
  .view-pane img { width: 100%; max-height: 300px; object-fit: contain; border-radius: 6px; background: #020617; border: 1px solid #1e293b; display: block; }
  .view-cap { font-size: 11px; color: #94a3b8; font-family: var(--mono); margin-top: 4px; }
  .player-ctrl { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
  .player-ctrl button { background: #2563eb; color: #fff; border: none; border-radius: 6px; padding: 5px 12px; font-size: 12px; font-weight: 600; cursor: pointer; }
  .player-ctrl input[type=range] { flex: 1; }
  .player-ctrl .time { font-family: var(--mono); font-size: 12px; color: #cbd5e1; min-width: 115px; text-align: right; }
  .thumbs { display: flex; gap: 6px; overflow-x: auto; margin-top: 8px; padding-bottom: 4px; }
  .thumbs img { height: 52px; border-radius: 4px; cursor: pointer; opacity: 0.65; border: 2px solid transparent; flex-shrink: 0; }
  .thumbs img.active { opacity: 1; border-color: #38bdf8; }

  /* Collapsible details (code, diffs, JSON) */
  details { margin-top: 10px; border: 1px solid rgba(15,23,42,0.12); border-radius: 8px; background: rgba(255,255,255,0.75); padding: 7px 11px; }
  details summary { cursor: pointer; font-size: 12.5px; font-weight: 700; color: var(--ink); }
  pre.code, .diff { margin: 8px 0 0; font-family: var(--mono); font-size: 11.5px; line-height: 1.45; background: #0f172a; color: #e2e8f0; padding: 10px 12px; border-radius: 7px; overflow-x: auto; max-height: 420px; white-space: pre-wrap; word-break: break-word; }
  .diff .add { color: #86efac; background: rgba(34,197,94,0.15); display: block; }
  .diff .del { color: #fca5a5; background: rgba(239,68,68,0.15); display: block; }
  .diff .hunk { color: #93c5fd; display: block; }
  .diff .ctx { color: #cbd5e1; display: block; }

  table { width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 8px; }
  th, td { text-align: left; padding: 7px 9px; border-bottom: 1px solid var(--border); }
  th { background: #f1f5f9; font-size: 11px; text-transform: uppercase; color: var(--muted); }
  code { font-family: var(--mono); font-size: 12px; background: rgba(15,23,42,0.07); padding: 1px 5px; border-radius: 4px; }
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-inner">
    <div class="top-row">
      <h1 id="hdr-title">MA-Task Harness 对话轨迹</h1>
      <div class="pill-row" id="hdr-pills"></div>
    </div>
    <div class="jumpbar" id="jumpbar"></div>
  </div>
</div>

<div class="chat-stream" id="chat-stream"></div>

<script>
const TRACE = __TRACE_JSON__;

function esc(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderDiff(diffText) {
  if (!diffText) return "";
  return '<div class="diff">' + diffText.split('\n').map(l => {
    if (l.startsWith('+++') || l.startsWith('---') || l.startsWith('@@')) return '<span class="hunk">' + esc(l) + '</span>';
    if (l.startsWith('+')) return '<span class="add">' + esc(l) + '</span>';
    if (l.startsWith('-')) return '<span class="del">' + esc(l) + '</span>';
    return '<span class="ctx">' + esc(l) + '</span>';
  }).join('') + '</div>';
}

const labels = TRACE.labels || {};
const rounds = TRACE.rounds || [];
const finalProbes = TRACE.final_probe_report || {};
const lastEngine = rounds.length ? ((rounds[rounds.length - 1].execution || {}).simulator_engine || "unknown") : "unknown";

// Top bar
document.getElementById("hdr-title").textContent = `MA-Task Harness · ${TRACE.title || TRACE.task}`;
document.getElementById("hdr-pills").innerHTML = [
  `<span class="pill ${TRACE.converged ? 'ok' : 'fail'}">${TRACE.converged ? '✓ 已收敛并冻结' : '✗ 未收敛'}</span>`,
  `<span class="pill blue">仿真器: ${esc(lastEngine === 'sapien_physx' ? '真实 SAPIEN + PhysX 250Hz' : lastEngine)}</span>`,
  `<span class="pill">Why: ${esc(labels.why)} · When: ${esc(labels.when)} · Who: ${esc(labels.who)}</span>`,
  `<span class="pill">共演化 ${rounds.length} 轮</span>`,
].join("");

const jumps = ['<a href="#sec-stage1">Stage 1: 需求澄清</a>'];
rounds.forEach(r => jumps.push(`<a href="#sec-r${r.round_index}">Round ${r.round_index} ${r.passed ? '✓' : '↻'}</a>`));
jumps.push('<a href="#sec-stage3">Stage 3: 探针认证 & 冻结包</a>');
document.getElementById("jumpbar").innerHTML = jumps.join("");

const parts = [];

// =========================================================================
// STAGE 1: User <-> Specifier Dialogue
// =========================================================================
parts.push('<div class="divider" id="sec-stage1">Stage 1 · 用户与 Specifier 多轮澄清 → TaskSpec.yaml</div>');
const s1 = TRACE.stage1;
if (s1 && s1.dialogue && s1.dialogue.length) {
  s1.dialogue.forEach(t => {
    const isUser = t.speaker === "user";
    let bodyHtml = "";
    if (t.questions && t.questions.length) {
      bodyHtml = (t.summary ? `<div style="margin-bottom:6px"><b>已确认：</b>${esc(t.summary)}</div>` : "") +
        `<div><b>待澄清问题：</b><ol>${t.questions.map(q => `<li>${esc(q)}</li>`).join("")}</ol></div>`;
    } else {
      bodyHtml = esc(t.content || "");
    }
    let extra = "";
    if (t.kind === "finalize" && t.validated) {
      extra = `<details><summary>📄 查看已校验的 TaskSpec.yaml（单点真值契约）</summary><pre class="code">${esc(TRACE.task_spec_yaml || "")}</pre></details>`;
    }
    if (t.validation_error) {
      extra += `<div class="badge fail" style="margin-top:8px">Schema 校验未通过（自动重试）: ${esc(t.validation_error)}</div>`;
    }
    parts.push(`
      <div class="row ${isUser ? 'right' : ''}">
        <div class="avatar ${isUser ? 'av-user' : 'av-spec'}">${isUser ? '用户' : 'S'}</div>
        <div class="bubble ${isUser ? 'user' : 'spec'}">
          <div class="b-head">
            <span class="role">${isUser ? '用户 (User)' : 'Specifier 规范代理'}</span>
            <span class="meta">Turn ${t.turn} · ${esc(t.kind)}${t.elapsed_s ? ' · ' + t.elapsed_s + 's' : ''}</span>
          </div>
          <div class="b-body">${bodyHtml}</div>
          ${extra}
        </div>
      </div>`);
  });
} else {
  parts.push(`
    <div class="sim-card">
      <div class="sim-title">📄 直接加载已有 TaskSpec.yaml</div>
      <details open><summary>查看 TaskSpec.yaml</summary><pre class="code">${esc(TRACE.task_spec_yaml || "")}</pre></details>
    </div>`);
}

// =========================================================================
// STAGE 2: Chronological Round-by-Round G^(n) <-> E^(n) <-> SAPIEN Simulator
// =========================================================================
rounds.forEach((r, rIdx) => {
  const ri = r.round_index;
  const audit = r.audit_report || {};
  const exec = r.execution || {};
  const phys = audit.physical || {};
  const miles = audit.milestones || {};
  const probes = audit.probes || {};
  const renders = exec.rendered_frames || [];

  parts.push(`<div class="divider" id="sec-r${ri}">Stage 2 · Round ${ri} 共演化循环（G^(${ri}) ↔ E^(${ri}) ↔ 仿真门禁）</div>`);

  // 1) Generator G^(n) Bubble
  const gDiffs = Object.entries(r.gen_diffs || {}).map(([fn, d]) =>
    `<div style="margin-top:8px"><code>${esc(fn)}</code>${renderDiff(d)}</div>`
  ).join("");
  parts.push(`
    <div class="row">
      <div class="avatar av-gen">G${ri}</div>
      <div class="bubble gen">
        <div class="b-head">
          <span class="role">Generator 生成代理 G^(${ri})（仅可读写 gen/）</span>
          <span class="meta">${esc(r.gen_turn.backend)} · ${r.gen_turn.elapsed_s}s · 修改 ${(r.gen_turn.files_changed || []).join(', ') || '无'}</span>
        </div>
        <div class="b-body">${esc(r.gen_turn.response || r.gen_turn.error || "(无文本输出)")}</div>
        ${gDiffs ? `<details><summary>📝 查看代码改动 Diff (${Object.keys(r.gen_diffs || {}).length} 个文件)</summary>${gDiffs}</details>` : ''}
        <details><summary>💬 查看发给隔离 G^(${ri}) 会话的原始 Prompt</summary><pre class="code">${esc(r.gen_turn.prompt || "")}</pre></details>
      </div>
    </div>`);

  // Optional G -> E Counter-Audit Bubble!
  if (r.counter_audit) {
    parts.push(`
      <div class="row">
        <div class="avatar av-gen" style="background:#ffe4e6;color:#be123c;border-color:#fb7185">反审</div>
        <div class="bubble counter">
          <div class="b-head">
            <span class="role" style="color:#be123c">⚡ G^(${ri}) → E^(${ri}) 反向审计（Counter-Audit）</span>
            <span class="meta">outbox/counter_audit.json</span>
          </div>
          <div class="b-body">生成代理发现物理轨迹本身合理，怀疑评测裁判存在误拒或 Schema 缺陷，已将反审证据投递至 Evaluator 收件箱：</div>
          <pre class="code">${esc(JSON.stringify(r.counter_audit, null, 2))}</pre>
        </div>
      </div>`);
  }

  // 2) Evaluator E^(n) Bubble (only show full bubble if it ran)
  if (r.eval_turn && r.eval_turn.ran) {
    const eDiffs = Object.entries(r.eval_diffs || {}).map(([fn, d]) =>
      `<div style="margin-top:8px"><code>${esc(fn)}</code>${renderDiff(d)}</div>`
    ).join("");
    parts.push(`
      <div class="row">
        <div class="avatar av-eval">E${ri}</div>
        <div class="bubble eval">
          <div class="b-head">
            <span class="role">Evaluator 评测代理 E^(${ri})（仅可读写 eval/，确定性无 LLM 裁判）</span>
            <span class="meta">${esc(r.eval_turn.backend)} · ${r.eval_turn.elapsed_s}s · 修改 ${(r.eval_turn.files_changed || []).join(', ') || '无'}</span>
          </div>
          <div class="b-body">${esc(r.eval_turn.response || r.eval_turn.error || "(无文本输出)")}</div>
          ${eDiffs ? `<details><summary>📝 查看评测器改动 Diff (${Object.keys(r.eval_diffs || {}).length} 个文件)</summary>${eDiffs}</details>` : ''}
          <details><summary>💬 查看发给隔离 E^(${ri}) 会话的原始 Prompt</summary><pre class="code">${esc(r.eval_turn.prompt || "")}</pre></details>
        </div>
      </div>`);
  }

  // 3) Authoritative Simulator + Deterministic Gate Execution Card
  const eng = exec.simulator_engine || "unknown";
  const badges = [
    `<span class="badge info">引擎: ${esc(eng === 'sapien_physx' ? 'SAPIEN + PhysX 250Hz' : eng)}</span>`,
    `<span class="badge ${phys.collision_free ? 'ok' : 'fail'}">${phys.collision_free ? '✓' : '✗'} 无非法碰撞</span>`,
    `<span class="badge ${phys.ik_continuous ? 'ok' : 'fail'}">${phys.ik_continuous ? '✓' : '✗'} IK 连续</span>`,
    `<span class="badge ${phys.grasps_persistent ? 'ok' : 'fail'}">${phys.grasps_persistent ? '✓' : '✗'} 抓持保持</span>`,
    `<span class="badge ${phys.terminal_at_rest ? 'ok' : 'fail'}">${phys.terminal_at_rest ? '✓' : '✗'} 终态静止</span>`,
    `<span class="badge ${miles.joint_success ? 'ok' : 'fail'}">S_role ${((miles.S_role||0)*100).toFixed(0)}% · S_collab ${((miles.S_collab||0)*100).toFixed(0)}% · SR ${miles.joint_success ? '100%' : '0%'}</span>`,
    `<span class="badge ${probes.matches_labels ? 'ok' : 'fail'}">${probes.matches_labels ? '✓' : '✗'} 探针认证</span>`,
  ].join("");

  let playerHtml = "";
  if (renders.length > 0) {
    const viewNames = Object.keys(renders[0].views || {});
    const thumbs = renders.map((rf, fi) => {
      const firstUrl = rf.views[viewNames[0]] || "";
      return `<img src="${firstUrl}" data-round="${rIdx}" data-frame="${fi}" class="${fi === 0 ? 'active' : ''}" title="t=${rf.t}s" />`;
    }).join("");
    playerHtml = `
      <div class="player-box" id="player-${rIdx}">
        <div class="player-views">
          ${viewNames.map(vn => `
            <div class="view-pane">
              <img id="img-${rIdx}-${esc(vn)}" src="${renders[0].views[vn] || ''}" alt="${esc(vn)}" />
              <div class="view-cap">真实 SAPIEN GPU 渲染视角: ${esc(vn)}</div>
            </div>`).join("")}
        </div>
        <div class="player-ctrl">
          <button type="button" data-play="${rIdx}">▶ 播放轨迹</button>
          <input type="range" min="0" max="${renders.length - 1}" value="0" data-scrub="${rIdx}" />
          <span class="time" id="time-${rIdx}">Frame 1/${renders.length} (t=${renders[0].t}s)</span>
        </div>
        <div class="thumbs">${thumbs}</div>
      </div>`;
  }

  const issues = [];
  if (audit.runtime_error) issues.push(`运行报错: ${audit.runtime_error}`);
  (audit.schema_errors || []).forEach(e => issues.push(`Schema 错误: ${e}`));
  (audit.evaluator_contract_errors || []).forEach(e => issues.push(`评测契约错误: ${e}`));
  (phys.reasons || []).forEach(e => issues.push(`物理门禁失败: ${e}`));
  if (miles.failed_predicates && miles.failed_predicates.length) {
    issues.push(`未通过里程碑 (${miles.failed_predicates.length}): ${miles.failed_predicates.join(', ')}`);
  }
  if (!probes.matches_labels && probes.certified) {
    issues.push(`探针未对齐声明标签: certified=${JSON.stringify(probes.certified)}`);
  }
  const issueHtml = issues.length
    ? `<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:10px 12px;font-size:13px;color:#991b1b;margin-bottom:8px"><b>本轮反馈给下一轮修复的诊断结论：</b><ul style="margin:6px 0 0;padding-left:18px">${issues.map(x => `<li>${esc(x)}</li>`).join("")}</ul></div>`
    : `<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:10px 12px;font-size:13px;color:#166534;margin-bottom:8px"><b>✓ 全部物理门禁、全部里程碑与 Appendix A 探针均通过！准备冻结任务包。</b></div>`;

  const flagRows = Object.entries(miles.flags || {}).map(([k, v]) =>
    `<tr><td><code>${esc(k)}</code></td><td><span class="badge ${v ? 'ok' : 'fail'}">${v ? 'PASS' : 'FAIL'}</span></td></tr>`
  ).join("");

  parts.push(`
    <div class="sim-card ${r.passed ? 'pass' : 'fail'}">
      <div class="sim-head">
        <div class="sim-title">🖥️ 独立子进程仿真与门禁裁决 — Round ${ri}</div>
        <span class="badge ${r.passed ? 'ok' : 'fail'}">${r.passed ? 'PASS · 收敛' : 'REJECT · 进入下一轮修复'}</span>
      </div>
      <div class="badges">${badges}</div>
      ${playerHtml}
      ${issueHtml}
      <details><summary>📋 查看 Round ${ri} 完整里程碑明细与 audit_report.json</summary>
        ${flagRows ? `<table><thead><tr><th>谓词 ID</th><th>状态</th></tr></thead><tbody>${flagRows}</tbody></table>` : ''}
        <pre class="code">${esc(JSON.stringify(audit, null, 2))}</pre>
      </details>
    </div>`);
});

// =========================================================================
// STAGE 3: Appendix A Probe Certification Matrix & Frozen Package
// =========================================================================
parts.push('<div class="divider" id="sec-stage3">Stage 3 · Appendix A 物理探针认证 & 冻结可复用任务包</div>');

const probeRows = (finalProbes.trials || []).map(t => `
  <tr>
    <td><code>${esc(t.mode)}</code></td>
    <td><span class="badge ${t.success ? 'ok' : 'fail'}">${t.success ? 'PASS' : 'FAIL'}</span></td>
    <td>${t.planner_feasible ? 'Yes' : 'No'}</td>
    <td>${t.physical_passed ? 'Yes' : 'No'}</td>
    <td>${t.goal_reached ? 'Yes' : 'No'}</td>
    <td style="font-size:12px;color:var(--muted)">${esc(t.reason || '—')}</td>
  </tr>`).join("");

const manifest = TRACE.package_manifest;
const pkgRows = manifest && manifest.files
  ? Object.entries(manifest.files).map(([fn, info]) =>
      `<tr><td><code>${esc(fn)}</code></td><td>${info.bytes} B</td><td><code>${esc(info.sha256.slice(0, 24))}...</code></td></tr>`
    ).join("")
  : '<tr><td colspan="3">未冻结任务包</td></tr>';

const finalCodeBlocks = Object.entries(TRACE.final_files || {}).map(([fn, code]) =>
  `<details><summary>📄 冻结源码: <code>${esc(fn)}</code></summary><pre class="code">${esc(code)}</pre></details>`
).join("");

parts.push(`
  <div class="sim-card ${TRACE.converged ? 'pass' : 'fail'}">
    <div class="sim-head">
      <div class="sim-title">🏆 Appendix A 操作性探针认证结果（Why / When / Who）</div>
      <span class="badge ${finalProbes.matches_labels ? 'ok' : 'fail'}">${finalProbes.matches_labels ? '✓ 三轴标签全部认证通过' : '✗ 存在标签不匹配'}</span>
    </div>
    <div class="badges">
      <span class="badge ${finalProbes.necessity_holds ? 'ok' : 'fail'}">Eq.(2) 多机必要性: ${finalProbes.necessity_holds ? '成立' : '未成立'}</span>
      <span class="badge ${finalProbes.why_certified === labels.why ? 'ok' : 'fail'}">Why: ${esc(finalProbes.why_certified)} (声明: ${esc(labels.why)})</span>
      <span class="badge ${finalProbes.when_certified === labels.when ? 'ok' : 'fail'}">When: ${esc(finalProbes.when_certified)} (声明: ${esc(labels.when)})</span>
      <span class="badge ${finalProbes.who_certified === labels.who ? 'ok' : 'fail'}">Who: ${esc(finalProbes.who_certified)} (声明: ${esc(labels.who)})</span>
    </div>
    <table>
      <thead><tr><th>探针模式</th><th>结论</th><th>规划可行</th><th>物理门禁</th><th>目标 V</th><th>仿真诊断说明</th></tr></thead>
      <tbody>${probeRows}</tbody>
    </table>
    <ul style="margin:10px 0 0;padding-left:18px;font-size:13px;color:var(--muted)">
      ${(finalProbes.notes || []).map(n => `<li>${esc(n)}</li>`).join("")}
    </ul>
  </div>
  <div class="sim-card">
    <div class="sim-head">
      <div class="sim-title">📦 冻结任务包清单（可独立复验 & 采数）</div>
    </div>
    <table><thead><tr><th>文件</th><th>大小</th><th>SHA-256</th></tr></thead><tbody>${pkgRows}</tbody></table>
    ${finalCodeBlocks}
  </div>`);

document.getElementById("chat-stream").innerHTML = parts.join("\n");

// Wire interactive RGB playback controls for every round with real SAPIEN renders
const timers = {};
function setRoundFrame(rIdx, fIdx) {
  const renders = ((rounds[rIdx] || {}).execution || {}).rendered_frames || [];
  if (!renders.length) return;
  const rf = renders[Math.max(0, Math.min(fIdx, renders.length - 1))];
  Object.entries(rf.views || {}).forEach(([vn, url]) => {
    const img = document.getElementById(`img-${rIdx}-${vn}`);
    if (img) img.src = url;
  });
  const scrub = document.querySelector(`input[data-scrub="${rIdx}"]`);
  if (scrub) scrub.value = String(fIdx);
  const tm = document.getElementById(`time-${rIdx}`);
  if (tm) tm.textContent = `Frame ${fIdx + 1}/${renders.length} (t=${rf.t}s)`;
  document.querySelectorAll(`.thumbs img[data-round="${rIdx}"]`).forEach(el => {
    el.classList.toggle("active", Number(el.dataset.frame) === fIdx);
  });
}

document.querySelectorAll("input[data-scrub]").forEach(inp => {
  inp.addEventListener("input", () => setRoundFrame(Number(inp.dataset.scrub), Number(inp.value)));
});
document.querySelectorAll(".thumbs img").forEach(img => {
  img.addEventListener("click", () => setRoundFrame(Number(img.dataset.round), Number(img.dataset.frame)));
});
document.querySelectorAll("button[data-play]").forEach(btn => {
  btn.addEventListener("click", () => {
    const rIdx = Number(btn.dataset.play);
    const renders = ((rounds[rIdx] || {}).execution || {}).rendered_frames || [];
    if (!renders.length) return;
    if (timers[rIdx]) {
      clearInterval(timers[rIdx]);
      timers[rIdx] = null;
      btn.textContent = "▶ 播放轨迹";
    } else {
      btn.textContent = "⏸ 暂停";
      timers[rIdx] = setInterval(() => {
        const scrub = document.querySelector(`input[data-scrub="${rIdx}"]`);
        const cur = scrub ? Number(scrub.value) : 0;
        setRoundFrame(rIdx, (cur + 1) % renders.length);
      }, 260);
    }
  });
});
</script>
</body>
</html>
"""


def render_trace_html(trace_json_path: Path, out_html_path: Path) -> Path:
    """Render `harness_trace.json` as a unified chronological dialogue HTML report."""
    data: Dict[str, Any] = json.loads(Path(trace_json_path).read_text(encoding="utf-8"))
    embedded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = _CHAT_HTML_TEMPLATE.replace(
        "__TITLE__", str(data.get("title", data.get("task", "Custom Task")))
    ).replace("__TRACE_JSON__", embedded)
    out = Path(out_html_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
