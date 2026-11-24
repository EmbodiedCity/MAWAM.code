"""Stage (1) Specifier: multi-round Why–When–Who clarification -> validated TaskSpec.yaml."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ma_task_harness.orchestrator.backends import CodingBackend
from ma_task_harness.spec.schema import (
    ROBOT_PRESETS,
    TaskSpec,
    TaskSpecValidationError,
    load_task_spec,
    load_task_spec_from_text,
    save_task_spec,
)

AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"
_YAML_FENCE_RE = re.compile(r"```(?:yaml|yml)\s*\n(.*?)\n```", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(\{.*?\})\n```", re.DOTALL)


def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    m = _JSON_FENCE_RE.search(text)
    candidates = [m.group(1)] if m else []
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        candidates.append(s)
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        candidates.append(s[start : end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _extract_yaml(text: str) -> Optional[str]:
    m = _YAML_FENCE_RE.search(text)
    if m:
        return m.group(1)
    if "task:" in text and "team:" in text and "labels:" in text:
        return text
    return None


def _robot_catalog_text(custom_catalog: Optional[Dict[str, Any]] = None) -> str:
    catalog = dict(ROBOT_PRESETS)
    if custom_catalog:
        catalog.update(custom_catalog)
    return json.dumps(catalog, indent=2)


def run_specify(
    idea: str,
    backend: CodingBackend,
    work_root: Path,
    clarification_answers: Optional[Sequence[str]] = None,
    interactive: bool = False,
    max_clarify_rounds: int = 2,
    max_repair_retries: int = 3,
    custom_robot_catalog: Optional[Dict[str, Any]] = None,
) -> Tuple[TaskSpec, Dict[str, Any]]:
    """Run Stage 1 multi-round Why–When–Who clarification and return `(TaskSpec, stage1_trace)`."""
    work_root = Path(work_root).resolve()
    ws = work_root / "stage1_specifier"
    ws.mkdir(parents=True, exist_ok=True)
    sys_prompt = (AGENTS_DIR / "SPECIFIER.md").read_text(encoding="utf-8")

    # Provide one concise example TaskSpec structure (schema reference only, not a restriction)
    schema_hint = """Required TaskSpec.yaml top-level keys:
  task: <snake_case_slug>
  title: "<Human Title>"
  version: v1
  description: "<Concise physical summary>"
  nominal_duration_s: 30.0
  control_hz: 25.0
  physics_hz: 250.0
  team:
    A: {robot: <name>, arms: <1|2>, action_dim: <8|14|16>, views: [...], workspace_bounds: [[x0,y0,z0],[x1,y1,z1]], observable_zones: [...], payload_kg: <float>}
    B: ...  # supports N >= 2 agents and arbitrary robot types
  labels: {why: information|spatial|capability, when: asynchronous|partial|synchronous, who: homogeneous|heterogeneous}
  prompts: {global: "...", A: "...", B: "..."}
  thresholds: {name: float, ...}
  scene_params: { ... physical geometry/stiffness/mass/layout ... }
  goal: [{id: goal_..., kind: <snake_case>, description: "...", params: {...}}]
  role_milestones:
    A: [{id: m_A1, kind: <snake_case>, description: "...", params: {...}}, ...]
    B: [{id: m_B1, kind: <snake_case>, description: "...", params: {...}}, ...]
  collab_milestones: [{id: c_1, kind: <snake_case>, description: "...", params: {...}}, ...]
  probes:
    necessity: [single_A, single_B, relax_O, relax_W, relax_U]  # include single_<aid> for every agent
    temporal: {serialize: true, delay_seconds: [0.08, 0.8, 2.5], sync_break_tolerance_s: 0.20}
    embodiment: [role_swap]  # required if heterogeneous, [] if homogeneous
"""

    dialogue: List[Dict[str, Any]] = [
        {"turn": 0, "speaker": "user", "kind": "idea", "content": idea}
    ]
    answers_queue = list(clarification_answers or [])
    clarify_rounds_done = 0

    # --- Clarification turns ---
    while clarify_rounds_done < max_clarify_rounds and (interactive or answers_queue):
        history_text = "\n\n".join(
            f"[{d['speaker'].upper()} turn {d['turn']}]: {d['content']}" for d in dialogue
        )
        prompt = (
            f"Available robot presets (you may also define custom robots):\n{_robot_catalog_text(custom_robot_catalog)}\n\n"
            f"Conversation so far:\n{history_text}\n\n"
            f"Clarification round {clarify_rounds_done + 1} of {max_clarify_rounds}.\n"
            "Identify the most important remaining ambiguities along Why (multi-agent necessity & O/W/U), "
            "When (asynchronous / partial / synchronous), Who (team embodiments & roles), and concrete physical geometry.\n"
            "Do NOT write TaskSpec.yaml yet. Respond ONLY with a JSON object:\n"
            '{"status": "clarify", "summary": "...", "questions": ["Q1?", "Q2?", "Q3?"]}'
        )
        res = backend.run(sys_prompt, prompt, ws)
        if not res.ok:
            raise RuntimeError(f"Specifier clarification round failed: {res.error}")
        parsed = _extract_json_obj(res.stdout) or {
            "status": "clarify",
            "summary": "",
            "questions": [res.stdout.strip()],
        }
        dialogue.append(
            {
                "turn": len(dialogue),
                "speaker": "specifier",
                "kind": "clarify",
                "elapsed_s": res.elapsed_s,
                "content": res.stdout.strip(),
                "summary": parsed.get("summary", ""),
                "questions": parsed.get("questions", []),
            }
        )

        if interactive:
            print("\n[Specifier questions]")
            for qi, q in enumerate(parsed.get("questions", []), 1):
                print(f"  {qi}. {q}")
            ans = input("\nYour clarification answer (empty to finalize): ").strip()
            if not ans:
                break
        else:
            ans = answers_queue.pop(0)

        dialogue.append({"turn": len(dialogue), "speaker": "user", "kind": "answer", "content": ans})
        clarify_rounds_done += 1
        print(f"[Stage 1] Clarify round {clarify_rounds_done}/{max_clarify_rounds} done ({res.elapsed_s}s, {len(parsed.get('questions', []))} questions)", flush=True)

    # --- Finalize & validate TaskSpec.yaml ---
    spec_file = ws / "TaskSpec.yaml"
    spec: Optional[TaskSpec] = None
    validation_error: Optional[str] = None

    for attempt in range(1, max_repair_retries + 1):
        history_text = "\n\n".join(
            f"[{d['speaker'].upper()} turn {d['turn']}]: {d['content']}" for d in dialogue
        )
        err_block = (
            f"\n\nPREVIOUS VALIDATION ERROR (fix this cleanly):\n{validation_error}\n"
            if validation_error
            else ""
        )
        prompt = (
            f"Available robot presets:\n{_robot_catalog_text(custom_robot_catalog)}\n\n"
            f"{schema_hint}\n"
            f"Full User <-> Specifier clarification transcript:\n{history_text}"
            f"{err_block}\n\n"
            "Now finalize and WRITE `TaskSpec.yaml` directly into the current workspace directory "
            "(and also briefly confirm the Why / When / Who labels and key thresholds in your text reply).\n"
            "Make sure:\n"
            "  1. `labels.who` matches the intrinsic robot descriptors in `team`.\n"
            "  2. Every `<threshold>` placeholder in `goal`, `role_milestones`, `collab_milestones` is defined in `thresholds`.\n"
            "  3. `probes.necessity` includes `single_<aid>` for every agent followed by `relax_O`, `relax_W`, `relax_U`.\n"
            "  4. If heterogeneous, `probes.embodiment` includes `role_swap`.\n"
            "  5. `scene_params` includes concrete numeric positions/dimensions/physics constants so G^(n) and E^(n) share unambiguous geometry."
        )
        res = backend.run(sys_prompt, prompt, ws)
        if not res.ok:
            raise RuntimeError(f"Specifier finalize failed: {res.error}")

        try:
            if spec_file.exists():
                spec = load_task_spec(spec_file)
            else:
                yml = _extract_yaml(res.stdout)
                if not yml:
                    raise TaskSpecValidationError("Specifier did not write TaskSpec.yaml or output a YAML block.")
                spec = load_task_spec_from_text(yml)
                save_task_spec(spec, spec_file)
            dialogue.append(
                {
                    "turn": len(dialogue),
                    "speaker": "specifier",
                    "kind": "finalize",
                    "attempt": attempt,
                    "elapsed_s": res.elapsed_s,
                    "validated": True,
                    "content": res.stdout.strip(),
                    "task_sha256": spec.sha256(),
                }
            )
            break
        except (TaskSpecValidationError, ValueError, KeyError) as exc:
            validation_error = str(exc)
            dialogue.append(
                {
                    "turn": len(dialogue),
                    "speaker": "specifier",
                    "kind": "finalize_retry",
                    "attempt": attempt,
                    "elapsed_s": res.elapsed_s,
                    "validated": False,
                    "validation_error": validation_error,
                    "content": res.stdout.strip(),
                }
            )

    if spec is None:
        raise TaskSpecValidationError(f"Specifier failed to produce a valid TaskSpec after {max_repair_retries} attempts: {validation_error}")

    final_spec_path = work_root / "TaskSpec.yaml"
    save_task_spec(spec, final_spec_path)
    stage1_trace = {
        "stage": 1,
        "idea": idea,
        "backend": backend.name,
        "task": spec.task,
        "title": spec.title,
        "labels": spec.labels.to_dict(),
        "task_sha256": spec.sha256(),
        "task_spec_path": str(final_spec_path),
        "task_spec_yaml": final_spec_path.read_text(encoding="utf-8"),
        "dialogue": dialogue,
    }
    (work_root / "stage1_trace.json").write_text(
        json.dumps(stage1_trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return spec, stage1_trace
