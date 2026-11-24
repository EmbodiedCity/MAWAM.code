# MA-Task Harness

Task-agnostic coding-agent co-evolution framework from **MAWAM** (Section 3.1–3.2 & Appendix A–B).

Converts **any** user-proposed multi-robot task idea ($N \ge 2$ homogeneous or heterogeneous robots, built-in or custom embodiments, open-ended predicates) into a verified, reusable task package in three stages:

1. **Stage 1 — Specify (`specify`)**: Multi-round *Why–When–Who* clarification between User and Specifier sub-agent $\to$ validated `TaskSpec.yaml`.
2. **Stage 2 — Co-Evolve (`coevolve`)**: Isolated code-writing sub-agents $G^{(n)}$ (owns `gen/scene.py`, `gen/expert.py`) and $E^{(n)}$ (owns deterministic, LLM-free `eval/evaluator.py`) iteratively repair each other via `audit_report.json` ($E \to G$) and `counter_audit.json` ($G \to E$), verified each round in an isolated subprocess against universal physical gates (i)–(iv), Eq. (4)–(5) milestone contracts, and Appendix A operational probes.
3. **Stage 3 — Certify & Freeze (`verify-pkg`, `render-trace`)**: Freezes a SHA-256 verified standalone package (`TaskSpec.yaml`, `gen/`, `eval/`, `probe_certificate.json`, `run_verify.py`) and renders a standalone interactive HTML trace viewer (`trace_viewer.html`).

The 7 MA-Bench task specifications live under `examples/mabench/` as reference examples — the harness imposes no restriction to those tasks.

## Quickstart

### 1. Offline CPU verification (no API key or GPU)

```bash
python3 ma_task_harness/tests/test_harness_generic.py
python3 -m ma_task_harness list-examples
python3 -m ma_task_harness validate
```

### 2. Author ANY brand-new custom multi-robot task (Live Claude Code / Codex / OpenAI API)

```bash
# Stage 1: Multi-round Why-When-Who clarification -> custom TaskSpec.yaml
python3 -m ma_task_harness specify \
  --backend claude --model sonnet \
  --work-dir ./runs/my_custom_task \
  --interactive \
  --idea "Your custom N-robot collaborative manipulation task idea..."

# 2. Stage 2 & 3: Real SAPIEN + PhysX + mplib G^(n) <-> E^(n) code co-evolution + probes + freeze + dialogue HTML
ROBOTWIN_ROOT=./third_party/RoboTwin \
python3 -m ma_task_harness coevolve \
  --spec ./runs/my_custom_task/TaskSpec.yaml \
  --backend claude --model sonnet \
  --sim-backend sapien \
  --sim-python /path/to/sapien_env/bin/python \
  --work-dir ./runs/my_custom_task \
  --max-rounds 10 \
  --render-html ./runs/my_custom_task/trace_dialogue.html

# Independently re-verify the frozen task package on real SAPIEN + PhysX
python3 -m ma_task_harness verify-pkg ./runs/my_custom_task/package \
  --sim-backend sapien --sim-python /path/to/sapien_env/bin/python
```

Switch `--backend` among `claude`, `codex`, or `openai` (any OpenAI-compatible API via `OPENAI_API_KEY` / `OPENAI_BASE_URL`). Pass `--sim-python /path/to/sapien_env/bin/python` to execute generated code inside a RoboTwin 2.0 / SAPIEN environment.

## Layout

```text
ma_task_harness/
├── spec/                  # Task-agnostic N-agent TaskSpec schema & YAML loader
├── runtime/               # State types, N-agent capability relaxations, RoboTwin bridge, isolated subprocess runner
├── gates/                 # Universal physical checks (i)-(iv), Eq.(4-5) scorer, Appendix A probe arbiter
├── agents/                # SPECIFIER / GENERATOR / EVALUATOR charters, generic skills, API templates
├── orchestrator/          # Pluggable backends (claude|codex|openai|callable), Stage 1 specifier, Stage 2-3 driver, CLI
├── viz/                   # Zero-CDN standalone interactive HTML trace viewer (`render_trace.py`)
├── examples/mabench/      # The 7 MA-Bench TaskSpec examples
└── tests/                 # Task-agnostic N=3 & full 3-stage co-evolution unit tests
```
