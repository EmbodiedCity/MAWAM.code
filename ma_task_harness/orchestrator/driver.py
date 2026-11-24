"""Stage (2) & (3) Task-Agnostic Co-Evolution Driver (Paper Section 3.2).

Orchestrates genuinely isolated, code-writing sub-agents:
  - Generator G^(n): owns `gen/` (`scene.py`, `expert.py`), reads `TaskSpec.yaml`
    and `inbox/audit_report.json`, emits optional `outbox/counter_audit.json`
    or `outbox/spec_revision.json`.
  - Evaluator E^(n): owns `eval/` (`evaluator.py`), reads `TaskSpec.yaml`,
    `inbox/audit_report.json`, and `inbox/counter_audit.json`.
  - Authoritative deterministic subprocess runner executes `gen/expert.py` +
    `eval/evaluator.py` + universal `PhysicalGate` + Appendix A `certify_probes`.
  - Freezes a self-contained runnable package in `package/` ONLY when every gate,
    every milestone (S_k=1, S_collab=1, SR=1), and every probe certifies the spec.
  - Records the complete unedited trace (dialogue, prompts, agent responses,
    code diffs, audits, counter-audits, probe tables, physics keyframes) to
    `harness_trace.json` for auditability and HTML visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional

from ma_task_harness.orchestrator.backends import CodingBackend
from ma_task_harness.runtime.subprocess_runner import run_candidate_subprocess
from ma_task_harness.spec.schema import TaskSpec, load_task_spec, save_task_spec

AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"


def _read_agent_file(rel: str) -> str:
    p = AGENTS_DIR / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _snapshot_py_dir(directory: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not directory.is_dir():
        return out
    for p in sorted(directory.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        out[str(p.relative_to(directory))] = p.read_text(encoding="utf-8")
    return out


def _unified_diffs(before: Dict[str, str], after: Dict[str, str], prefix: str) -> Dict[str, str]:
    diffs: Dict[str, str] = {}
    for key in sorted(set(before) | set(after)):
        old_txt = before.get(key, "")
        new_txt = after.get(key, "")
        if old_txt == new_txt:
            continue
        diff_lines = list(
            difflib.unified_diff(
                old_txt.splitlines(keepends=True),
                new_txt.splitlines(keepends=True),
                fromfile=f"a/{prefix}/{key}",
                tofile=f"b/{prefix}/{key}",
                n=3,
            )
        )
        diffs[f"{prefix}/{key}"] = "".join(diff_lines)
    return diffs


def _sync_py_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class RoundRecord:
    round_index: int
    timestamp: float
    gen_turn: Dict[str, Any]
    eval_turn: Dict[str, Any]
    execution: Dict[str, Any]
    audit_report: Dict[str, Any]
    counter_audit: Optional[Dict[str, Any]] = None
    spec_revision: Optional[Dict[str, Any]] = None
    gen_diffs: Dict[str, str] = field(default_factory=dict)
    eval_diffs: Dict[str, str] = field(default_factory=dict)
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_index": self.round_index,
            "timestamp": self.timestamp,
            "gen_turn": self.gen_turn,
            "eval_turn": self.eval_turn,
            "execution": self.execution,
            "audit_report": self.audit_report,
            "counter_audit": self.counter_audit,
            "spec_revision": self.spec_revision,
            "gen_diffs": self.gen_diffs,
            "eval_diffs": self.eval_diffs,
            "passed": self.passed,
        }


@dataclass
class CoEvolutionResult:
    task: str
    title: str
    converged: bool
    rounds: List[RoundRecord]
    probe_report: Dict[str, Any]
    package_dir: Optional[str] = None
    trace_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "title": self.title,
            "converged": self.converged,
            "rounds": [r.to_dict() for r in self.rounds],
            "probe_report": self.probe_report,
            "package_dir": self.package_dir,
            "trace_path": self.trace_path,
        }


class CoEvolutionDriver:
    """Task-agnostic dual-agent code-coevolution orchestrator (G^(n) <-> E^(n))."""

    def __init__(
        self,
        spec: TaskSpec,
        backend: CodingBackend,
        work_root: Path,
        max_rounds: int = 5,
        sim_backend: str = "any",
        sim_python: Optional[str] = None,
        sim_ssh: Optional[str] = None,
        sim_ssh_config: Optional[str] = None,
        sim_env_script: Optional[str] = None,
        sim_gpu: str = "0",
        eval_backend: Optional[CodingBackend] = None,
    ) -> None:
        self.spec = spec
        self.gen_backend = backend
        self.eval_backend = eval_backend or backend
        self.work_root = Path(work_root).resolve()
        self.max_rounds = int(max_rounds)
        self.sim_backend = str(sim_backend).strip().lower()
        self.sim_python = sim_python
        self.sim_ssh = sim_ssh
        self.sim_ssh_config = sim_ssh_config
        self.sim_env_script = sim_env_script
        self.sim_gpu = str(sim_gpu)

        self.spec_path = self.work_root / "TaskSpec.yaml"
        self.gen_root = self.work_root / "gen"
        self.eval_root = self.work_root / "eval"

    def _prepare_role_workspace(self, role: str, n: int) -> Path:
        ws = self.work_root / f"round_{n:02d}" / role
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "inbox").mkdir(exist_ok=True)
        (ws / "outbox").mkdir(exist_ok=True)
        shutil.copy2(self.spec_path, ws / "TaskSpec.yaml")
        # Provide exact runtime & schema API signatures locally inside the isolated workspace
        # so sub-agents never need to search outside their workspace directory.
        rt_dir = Path(__file__).resolve().parent.parent / "runtime"
        (ws / "API_REFERENCE.py").write_text((rt_dir / "state_types.py").read_text(encoding="utf-8"), encoding="utf-8")
        if self.sim_backend == "sapien" and role == "generator":
            (ws / "SAPIEN_WORLD_API.py").write_text(
                (rt_dir / "sapien_world.py").read_text(encoding="utf-8"), encoding="utf-8"
            )
        skill_txt = _read_agent_file("skills/multi_agent_physics.md")
        if skill_txt:
            (ws / "PHYSICS_SKILLS.md").write_text(skill_txt, encoding="utf-8")
        if role == "generator":
            (ws / "gen").mkdir(exist_ok=True)
            _sync_py_dir(self.gen_root, ws / "gen")
            (ws / "expert_template.py").write_text(
                _read_agent_file("templates/expert_template.py"), encoding="utf-8"
            )
        else:
            (ws / "eval").mkdir(exist_ok=True)
            _sync_py_dir(self.eval_root, ws / "eval")
            (ws / "evaluator_template.py").write_text(
                _read_agent_file("templates/evaluator_template.py"), encoding="utf-8"
            )
        return ws

    def _build_audit_report(self, n: int, execution: Dict[str, Any]) -> Dict[str, Any]:
        phys = execution.get("physical", {})
        miles = execution.get("milestones", {})
        probes = execution.get("probe_report", {})
        schema_errs = execution.get("schema_errors", [])
        contract_errs = miles.get("contract_errors", [])
        rt_err = execution.get("runtime_error")

        failed_milestones = [k for k, v in miles.get("flags", {}).items() if not v]
        failed_probes = [
            {"mode": t["mode"], "success": t["success"], "reason": t.get("reason", "")}
            for t in probes.get("trials", [])
        ]
        passed = bool(
            execution.get("ok")
            and not rt_err
            and not schema_errs
            and not contract_errs
            and phys.get("passed", False)
            and miles.get("joint_success", False)
            and probes.get("matches_labels", False)
        )
        return {
            "round": n,
            "passed": passed,
            "runtime_error": rt_err,
            "traceback": (execution.get("traceback") or "")[-2000:],
            "schema_errors": schema_errs,
            "evaluator_contract_errors": contract_errs,
            "physical": {
                "passed": phys.get("passed", False),
                "collision_free": phys.get("collision_free", False),
                "ik_continuous": phys.get("ik_continuous", False),
                "grasps_persistent": phys.get("grasps_persistent", False),
                "terminal_at_rest": phys.get("terminal_at_rest", False),
                "reasons": phys.get("reasons", []),
            },
            "milestones": {
                "S_role": miles.get("S_role", 0.0),
                "S_collab": miles.get("S_collab", 0.0),
                "goal_reached": miles.get("goal_reached", False),
                "joint_success": miles.get("joint_success", False),
                "failed_predicates": failed_milestones,
                "flags": miles.get("flags", {}),
                "notes": miles.get("notes", {}),
            },
            "probes": {
                "necessity_holds": probes.get("necessity_holds", False),
                "declared": probes.get("declared_labels", {}),
                "certified": {
                    "why": probes.get("why_certified"),
                    "when": probes.get("when_certified"),
                    "who": probes.get("who_certified"),
                },
                "matches_labels": probes.get("matches_labels", False),
                "notes": probes.get("notes", []),
                "trials": failed_probes,
            },
            "team_trace_stats": execution.get("team_trace_stats", {}),
        }

    def run_coevolution(
        self,
        initial_gen_dir: Optional[Path] = None,
        initial_eval_dir: Optional[Path] = None,
        resume: bool = False,
    ) -> CoEvolutionResult:
        """Run multi-round isolated G^(n) <-> E^(n) code co-evolution until all checks & probes pass."""
        self.work_root.mkdir(parents=True, exist_ok=True)
        save_task_spec(self.spec, self.spec_path)
        self.gen_root.mkdir(exist_ok=True)
        self.eval_root.mkdir(exist_ok=True)
        if initial_gen_dir and Path(initial_gen_dir).is_dir():
            _sync_py_dir(Path(initial_gen_dir), self.gen_root)
        if initial_eval_dir and Path(initial_eval_dir).is_dir():
            _sync_py_dir(Path(initial_eval_dir), self.eval_root)

        gen_sys = _read_agent_file("GENERATOR.md")
        eval_sys = _read_agent_file("EVALUATOR.md")
        rounds: List[RoundRecord] = []
        last_audit: Optional[Dict[str, Any]] = None
        last_counter: Optional[Dict[str, Any]] = None
        converged = False
        package_dir: Optional[Path] = None
        start_round = 1

        if resume and (self.work_root / "harness_trace.json").exists():
            prev = json.loads((self.work_root / "harness_trace.json").read_text(encoding="utf-8"))
            for rdict in prev.get("rounds", []):
                rounds.append(
                    RoundRecord(
                        round_index=int(rdict["round_index"]),
                        timestamp=float(rdict.get("timestamp", 0.0)),
                        gen_turn=dict(rdict.get("gen_turn", {})),
                        eval_turn=dict(rdict.get("eval_turn", {})),
                        execution=dict(rdict.get("execution", {})),
                        audit_report=dict(rdict.get("audit_report", {})),
                        counter_audit=rdict.get("counter_audit"),
                        spec_revision=rdict.get("spec_revision"),
                        gen_diffs=dict(rdict.get("gen_diffs", {})),
                        eval_diffs=dict(rdict.get("eval_diffs", {})),
                        passed=bool(rdict.get("passed", False)),
                    )
                )
            if rounds:
                start_round = rounds[-1].round_index + 1
                last_audit = rounds[-1].audit_report
                last_counter = rounds[-1].counter_audit

        for n in range(start_round, self.max_rounds + 1):
            # --------------------------------------------------------------
            # 1. Generator G^(n) isolated session (writes gen/scene.py & gen/expert.py)
            # --------------------------------------------------------------
            gen_before = _snapshot_py_dir(self.gen_root)
            g_ws = self._prepare_role_workspace("generator", n)
            if last_audit is not None:
                (g_ws / "inbox" / "audit_report.json").write_text(
                    json.dumps(last_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )

            if n == 1 and not gen_before:
                if self.sim_backend == "sapien":
                    g_prompt = (
                        f"Round {n}: Read ONLY local workspace files `TaskSpec.yaml`, `SAPIEN_WORLD_API.py`, `API_REFERENCE.py`, and `PHYSICS_SKILLS.md`.\n"
                        f"Implement the REAL SAPIEN 3.0 + PhysX + mplib scene and multi-robot expert for custom task `{self.spec.task}` "
                        "by writing `gen/scene.py` and `gen/expert.py`.\n"
                        "CRITICAL REQUIREMENTS (REAL SAPIEN + PHYSX):\n"
                        "- You MUST use `SapienMultiRobotWorld` from `ma_task_harness.runtime.sapien_world` (exact API in `SAPIEN_WORLD_API.py`).\n"
                        "- Do NOT mock physics with NumPy! `rollout_expert(spec, mode)` MUST instantiate `SapienMultiRobotWorld(spec, mode)`, "
                        "build the real PhysX table/objects/fixtures/regions/robots (`add_table`, `add_box`, `add_spring_hinge_fixture`, `add_region`, `add_robot`), "
                        "add an `overview` camera (plus optional side camera), call `world.settle()`, execute robot motions via `world.move_arms_to(...)` / `world.set_gripper(...)` / `world.hold_steps(...)`, "
                        "and return `world.build_trace(...)`.\n"
                        "- Call `world.allow_grasp(arm_key, body_name, True)` while grasping an object and `False` after releasing so PhysX contact auditing distinguishes intended grasps from collisions.\n"
                        "- Check reachability / capabilities honestly via `effective_capabilities(spec, mode)` for ALL probe modes (`team`, `single_<aid>`, `relax_O`, `relax_W`, `relax_U`, `serialize`, `delay`, `role_swap`). "
                        "For infeasible single-agent / role-swap probe modes, return `EpisodeTrace(..., planner_feasible=False, planner_failure_reason=...)` immediately (or let PhysX show the failure).\n"
                        "- Tip: Use `world.move_tcp_to(arm_key, tcp_pos_xyz, steps=20, gripper=...)` for millimeter-accurate top-down grasping on Panda, UR5, and ALOHA (it automatically applies each URDF's `topdown_quat` and `tcp_z_offset`), and `world.add_camera('overview', ...)` + `world.add_camera('side', ...)` for dual-angle GPU renders.\n"
                        "- Keep `gen/scene.py` (~60-100 lines) and `gen/expert.py` (~120-200 lines) clean, concise, and fast.\n"
                        "Only edit files inside `gen/`. Summarize what you built concisely."
                    )
                else:
                    g_prompt = (
                        f"Round {n}: Read ONLY local workspace files `TaskSpec.yaml`, `API_REFERENCE.py`, `expert_template.py`, and `PHYSICS_SKILLS.md`.\n"
                        f"Implement the full multi-robot scene and expert planner for custom task `{self.spec.task}` "
                        "by writing `gen/scene.py` and `gen/expert.py`.\n"
                        "Requirements:\n"
                        "- `gen/expert.py` MUST export `rollout_expert(spec, mode) -> EpisodeTrace`.\n"
                        "- Import dataclasses from `ma_task_harness.runtime.state_types` (exact signatures in `API_REFERENCE.py`).\n"
                        "- Simulate genuine 250 Hz substeps (emitting 25 Hz ControlFrames), real contacts, continuous IK, "
                        "and realistic physical constraints (workspace reach, visibility, spring/gravity/tilt dynamics, arm/payload limits).\n"
                        "- Implement ALL probe modes cleanly via `effective_capabilities(spec, mode)`: `team`, `single_<aid>` "
                        "for every agent, `relax_O`, `relax_W`, `relax_U`, `serialize`, `delay` (`mode.delay_s`), and `role_swap`.\n"
                        "- Mark static/hinged fixtures with `is_fixture=True` and dynamic objects with `is_fixture=False`.\n"
                        "- Emit native SAPIEN `(3, 4)` camera extrinsic shapes.\n"
                        "- Keep `gen/scene.py` (~60-100 lines) and `gen/expert.py` (~140-220 lines) clean, concise, and fast to write.\n"
                        "Only edit files inside `gen/`. Summarize what you built concisely."
                    )
            else:
                g_prompt = (
                    f"Round {n}: Read `TaskSpec.yaml` and `inbox/audit_report.json`.\n"
                    "Diagnose why the previous round failed (runtime error, physical gate violation, missed milestone, or probe mismatch) "
                    "and repair `gen/scene.py` / `gen/expert.py`.\n"
                    "- If the failure is actually a false rejection, schema bug, or flawed detector in the evaluator while your physics "
                    "is valid, write a structured explanation to `outbox/counter_audit.json` (`{\"findings\": [...], \"evidence\": \"...\"}`).\n"
                    "- Never cheat probes or set `pose_overwritten=True`.\n"
                    "Only edit files inside `gen/` (and optionally write `outbox/counter_audit.json`)."
                )

            g_res = self.gen_backend.run(gen_sys, g_prompt, g_ws)
            _sync_py_dir(g_ws / "gen", self.gen_root)
            gen_after = _snapshot_py_dir(self.gen_root)
            gen_diffs = _unified_diffs(gen_before, gen_after, "gen")

            counter_audit: Optional[Dict[str, Any]] = None
            ca_file = g_ws / "outbox" / "counter_audit.json"
            if ca_file.exists():
                try:
                    counter_audit = json.loads(ca_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    counter_audit = {"raw": ca_file.read_text(encoding="utf-8")}
            spec_rev: Optional[Dict[str, Any]] = None
            sr_file = g_ws / "outbox" / "spec_revision.json"
            if sr_file.exists():
                try:
                    spec_rev = json.loads(sr_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    spec_rev = {"raw": sr_file.read_text(encoding="utf-8")}

            gen_turn = {
                "ok": g_res.ok,
                "backend": g_res.backend,
                "elapsed_s": g_res.elapsed_s,
                "prompt": g_prompt,
                "response": g_res.stdout.strip(),
                "error": g_res.error,
                "files_changed": sorted(gen_diffs.keys()),
            }

            # --------------------------------------------------------------
            # 2. Evaluator E^(n) isolated session (writes/repairs eval/evaluator.py)
            # --------------------------------------------------------------
            eval_before = _snapshot_py_dir(self.eval_root)
            prev_failed = (
                rounds[-2].audit_report.get("milestones", {}).get("failed_predicates", [])
                if len(rounds) >= 2
                else []
            )
            cur_failed = last_audit.get("milestones", {}).get("failed_predicates", []) if last_audit else []
            stalled_milestone = bool(
                last_audit is not None
                and last_audit.get("physical", {}).get("passed")
                and last_audit.get("milestones", {}).get("goal_reached")
                and cur_failed
                and cur_failed == prev_failed
            )
            need_eval = (
                (n == 1 and not eval_before)
                or (counter_audit is not None)
                or (last_audit is not None and bool(last_audit.get("schema_errors") or last_audit.get("evaluator_contract_errors")))
                or (last_audit is not None and bool(last_audit.get("runtime_error")) and "evaluator" in str(last_audit.get("traceback", "")).lower())
                or (
                    last_audit is not None
                    and bool(last_audit.get("milestones", {}).get("goal_reached"))
                    and float(last_audit.get("milestones", {}).get("S_role", 1.0)) == 0.0
                )
                or stalled_milestone
            )
            eval_diffs: Dict[str, str] = {}
            if need_eval:
                e_ws = self._prepare_role_workspace("evaluator", n)
                if last_audit is not None:
                    (e_ws / "inbox" / "audit_report.json").write_text(
                        json.dumps(last_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )
                if counter_audit is not None:
                    (e_ws / "inbox" / "counter_audit.json").write_text(
                        json.dumps(counter_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )

                if n == 1 and not eval_before:
                    e_prompt = (
                        f"Round {n}: Read ONLY local workspace files `TaskSpec.yaml`, `API_REFERENCE.py`, `evaluator_template.py`, and `PHYSICS_SKILLS.md`.\n"
                        f"Implement deterministic, strictly LLM-free evaluation for custom task `{self.spec.task}` "
                        "by writing `eval/evaluator.py`.\n"
                        "Requirements:\n"
                        "- Export `check_schema(spec, trace) -> List[str]` and `evaluate_task(spec, trace) -> Dict[str, Any]`.\n"
                        "- `evaluate_task` MUST return `{\"flags\": {<id>: bool, ...}, \"notes\": {...}}` covering EVERY predicate ID "
                        "in `spec.goal`, `spec.role_milestones`, and `spec.collab_milestones`, using thresholds resolved via `pred.resolve_params(spec.thresholds)`.\n"
                        "- Compute all predicates strictly from physical trace state (`frames`, `initial_bodies`, `bodies`, `contacts`, `arms`), "
                        "NEVER from expert phase strings.\n"
                        "- Accept both `(3, 4)` and `(4, 4)` camera extrinsic shapes in `check_schema`.\n"
                        "- Keep `eval/evaluator.py` clean and concise (~140-220 lines).\n"
                        "Only edit files inside `eval/`. Summarize your detectors concisely."
                    )
                else:
                    e_prompt = (
                        f"Round {n}: Read `TaskSpec.yaml`, `inbox/audit_report.json`, and (if present) `inbox/counter_audit.json`.\n"
                        "Repair any schema bug, missing predicate, contract error, or false-rejection detector defect in `eval/evaluator.py` "
                        "without weakening genuine physical or collaboration requirements.\n"
                        "Only edit files inside `eval/`."
                    )

                e_res = self.eval_backend.run(eval_sys, e_prompt, e_ws)
                _sync_py_dir(e_ws / "eval", self.eval_root)
                eval_after = _snapshot_py_dir(self.eval_root)
                eval_diffs = _unified_diffs(eval_before, eval_after, "eval")
                eval_turn = {
                    "ran": True,
                    "ok": e_res.ok,
                    "backend": e_res.backend,
                    "elapsed_s": e_res.elapsed_s,
                    "prompt": e_prompt,
                    "response": e_res.stdout.strip(),
                    "error": e_res.error,
                    "files_changed": sorted(eval_diffs.keys()),
                }
            else:
                eval_turn = {
                    "ran": False,
                    "ok": True,
                    "backend": self.eval_backend.name,
                    "elapsed_s": 0.0,
                    "prompt": "",
                    "response": "Evaluator unchanged this round (no evaluator defect or counter-audit).",
                    "error": "",
                    "files_changed": [],
                }

            # --------------------------------------------------------------
            # 3. Authoritative deterministic execution in isolated subprocess
            # --------------------------------------------------------------
            execution = run_candidate_subprocess(
                spec_path=self.spec_path,
                gen_dir=self.gen_root,
                eval_dir=self.eval_root,
                sim_python=self.sim_python,
                run_probes=True,
                require_sapien=(self.sim_backend == "sapien"),
                sim_ssh=self.sim_ssh,
                sim_ssh_config=self.sim_ssh_config,
                sim_env_script=self.sim_env_script,
                sim_gpu=self.sim_gpu,
            )
            audit_report = self._build_audit_report(n, execution)
            round_dir = self.work_root / f"round_{n:02d}"
            (round_dir / "audit_report.json").write_text(
                json.dumps(audit_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

            rec = RoundRecord(
                round_index=n,
                timestamp=time.time(),
                gen_turn=gen_turn,
                eval_turn=eval_turn,
                execution=execution,
                audit_report=audit_report,
                counter_audit=counter_audit,
                spec_revision=spec_rev,
                gen_diffs=gen_diffs,
                eval_diffs=eval_diffs,
                passed=bool(audit_report["passed"]),
            )
            rounds.append(rec)
            last_audit = audit_report
            last_counter = counter_audit
            print(
                f"[Stage 2] Round {n}/{self.max_rounds}: passed={rec.passed} "
                f"engine={execution.get('simulator_engine')} renders={len(execution.get('rendered_frames', []))} "
                f"phys={audit_report['physical'].get('passed')} "
                f"S_role={audit_report['milestones'].get('S_role', 0):.2f} "
                f"S_collab={audit_report['milestones'].get('S_collab', 0):.2f} "
                f"probes_ok={audit_report['probes'].get('matches_labels')} "
                f"gen={gen_turn['elapsed_s']}s eval={eval_turn['elapsed_s']}s",
                flush=True,
            )

            if rec.passed:
                converged = True
                package_dir = self.freeze_package(rounds, execution.get("probe_report", {}))
                break

        # Build unified harness_trace.json (merging Stage 1 if present)
        stage1_path = self.work_root / "stage1_trace.json"
        stage1_trace = json.loads(stage1_path.read_text(encoding="utf-8")) if stage1_path.exists() else None
        final_probes = rounds[-1].execution.get("probe_report", {}) if rounds else {}
        pkg_manifest = None
        if package_dir is not None and (package_dir / "MANIFEST.json").exists():
            pkg_manifest = json.loads((package_dir / "MANIFEST.json").read_text(encoding="utf-8"))

        trace_bundle = {
            "schema": "mawam.harness.trace.v2",
            "task": self.spec.task,
            "title": self.spec.title,
            "version": self.spec.version,
            "labels": self.spec.labels.to_dict(),
            "task_spec": self.spec.to_dict(),
            "task_spec_yaml": self.spec_path.read_text(encoding="utf-8"),
            "converged": converged,
            "total_rounds": len(rounds),
            "stage1": stage1_trace,
            "rounds": [r.to_dict() for r in rounds],
            "final_probe_report": final_probes,
            "package_dir": str(package_dir) if package_dir else None,
            "package_manifest": pkg_manifest,
            "final_files": {
                **{f"gen/{k}": v for k, v in _snapshot_py_dir(self.gen_root).items()},
                **{f"eval/{k}": v for k, v in _snapshot_py_dir(self.eval_root).items()},
            },
        }
        trace_path = self.work_root / "harness_trace.json"
        trace_path.write_text(json.dumps(trace_bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        return CoEvolutionResult(
            task=self.spec.task,
            title=self.spec.title,
            converged=converged,
            rounds=rounds,
            probe_report=final_probes,
            package_dir=str(package_dir) if package_dir else None,
            trace_path=str(trace_path),
        )

    def freeze_package(self, rounds: List[RoundRecord], probe_report: Dict[str, Any]) -> Path:
        """Stage (3): Freeze a self-contained, SHA-verified reusable task package."""
        pkg = self.work_root / "package"
        if pkg.exists():
            shutil.rmtree(pkg)
        pkg.mkdir(parents=True)
        save_task_spec(self.spec, pkg / "TaskSpec.yaml")
        _sync_py_dir(self.gen_root, pkg / "gen")
        _sync_py_dir(self.eval_root, pkg / "eval")
        (pkg / "probe_certificate.json").write_text(
            json.dumps(probe_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        verify_script = '''#!/usr/bin/env python3
"""Standalone verification script for this frozen MA-Task Harness package."""
from pathlib import Path
import json
from ma_task_harness.runtime.subprocess_runner import execute_round_in_process

ROOT = Path(__file__).resolve().parent
res = execute_round_in_process(ROOT / "TaskSpec.yaml", ROOT / "gen", ROOT / "eval", run_probes=True)
ok = bool(res["ok"] and res["physical"]["passed"] and res["milestones"]["joint_success"] and res["probe_report"]["matches_labels"])
print(json.dumps({
    "task": res["probe_report"].get("declared_labels"),
    "physical_passed": res["physical"].get("passed"),
    "S_role": res["milestones"].get("S_role"),
    "S_collab": res["milestones"].get("S_collab"),
    "joint_success": res["milestones"].get("joint_success"),
    "probes_certified": res["probe_report"].get("matches_labels"),
    "all_passed": ok,
}, indent=2))
raise SystemExit(0 if ok else 1)
'''
        (pkg / "run_verify.py").write_text(verify_script, encoding="utf-8")
        (pkg / "README.md").write_text(
            f"# Frozen Task Package: {self.spec.title}\n\n"
            f"- Task slug: `{self.spec.task}` (`{self.spec.version}`)\n"
            f"- Certified labels: Why=`{self.spec.labels.why}`, When=`{self.spec.labels.when}`, Who=`{self.spec.labels.who}`\n"
            f"- Team: `{', '.join(f'{k}:{v.robot}({v.arms}-arm)' for k, v in self.spec.team.items())}`\n"
            f"- Co-evolution rounds: `{len(rounds)}`\n"
            f"- Re-verify anytime: `python3 run_verify.py`\n",
            encoding="utf-8",
        )

        files_sha = {}
        for p in sorted(pkg.rglob("*")):
            if p.is_file() and p.name != "MANIFEST.json":
                files_sha[str(p.relative_to(pkg))] = {"bytes": p.stat().st_size, "sha256": _sha256(p)}
        manifest = {
            "task": self.spec.task,
            "title": self.spec.title,
            "version": self.spec.version,
            "labels": self.spec.labels.to_dict(),
            "rounds_to_converge": len(rounds),
            "probe_matches_labels": bool(probe_report.get("matches_labels", False)),
            "files": files_sha,
        }
        (pkg / "MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return pkg
