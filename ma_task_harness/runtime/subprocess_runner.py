"""Isolated subprocess runner (local or remote GPU via SSH) for `gen/` + `eval/`.

Features:
  - Enforces `--sim-backend sapien` by verifying `trace.metadata["simulator_engine"] == "sapien_physx"`
    so Generator G^(n) cannot bypass real SAPIEN + PhysX with mocked NumPy physics
  - Preserves real GPU-rendered SAPIEN RGB camera frames (`rendered_frames`) in the JSON result
  - Supports both local execution (`--sim-python`) and transparent remote GPU execution
    (`--sim-ssh`, `--sim-ssh-config`, `--sim-env-script`) without hardcoding any private host
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import traceback
from typing import Any, Dict, List, Optional

from ma_task_harness.gates.physical import PhysicalGate
from ma_task_harness.gates.probes import ProbeTrial, certify_probes
from ma_task_harness.gates.scoring import aggregate_evaluator_output
from ma_task_harness.runtime.state_types import EpisodeTrace, ExecutionMode
from ma_task_harness.spec.schema import TaskSpec, load_task_spec


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _eval_one_mode(
    spec: TaskSpec,
    expert_mod: Any,
    eval_mod: Any,
    phys_gate: PhysicalGate,
    mode: ExecutionMode,
    require_sapien: bool = False,
) -> tuple[EpisodeTrace, Dict[str, Any], Dict[str, Any], List[str]]:
    trace: EpisodeTrace = expert_mod.rollout_expert(spec, mode)
    if not isinstance(trace, EpisodeTrace):
        raise TypeError(f"rollout_expert must return EpisodeTrace, got {type(trace).__name__}")

    schema_errors: List[str] = []
    if require_sapien and trace.frames:
        eng = str(trace.metadata.get("simulator_engine", ""))
        if eng != "sapien_physx":
            schema_errors.append(
                f"sim_backend=sapien requires real SAPIEN+PhysX (`SapienMultiRobotWorld`), "
                f"got simulator_engine={eng!r}. Do not mock physics with NumPy."
            )

    phys = phys_gate.evaluate(trace)
    if hasattr(eval_mod, "check_schema"):
        res = eval_mod.check_schema(spec, trace)
        if isinstance(res, list):
            schema_errors.extend(str(x) for x in res)

    if trace.planner_feasible and trace.frames:
        raw_eval = eval_mod.evaluate_task(spec, trace)
        miles = aggregate_evaluator_output(spec, raw_eval)
    else:
        miles = aggregate_evaluator_output(
            spec,
            {
                "flags": {
                    **{g.id: False for g in spec.goal},
                    **{m.id: False for ml in spec.role_milestones.values() for m in ml},
                    **{c.id: False for c in spec.collab_milestones},
                },
                "notes": {"planner_failure": trace.planner_failure_reason or "infeasible"},
            },
        )
    return trace, phys.to_dict(), miles.to_dict(), schema_errors


def _run_relax_probe(
    spec: TaskSpec,
    expert_mod: Any,
    eval_mod: Any,
    phys_gate: PhysicalGate,
    relax_mode: str,
    require_sapien: bool = False,
) -> ProbeTrial:
    last_reason = ""
    for aid in spec.agent_ids:
        em = ExecutionMode(mode=relax_mode, primary_agent=aid, seed=0)
        trace, phys, miles, s_errs = _eval_one_mode(
            spec, expert_mod, eval_mod, phys_gate, em, require_sapien=require_sapien
        )
        ok = bool(trace.planner_feasible and phys["passed"] and miles["goal_reached"] and not s_errs)
        if ok:
            return ProbeTrial(
                mode=relax_mode,
                success=True,
                planner_feasible=True,
                physical_passed=True,
                goal_reached=True,
                joint_success=bool(miles["joint_success"]),
                S_role=float(miles["S_role"]),
                S_collab=float(miles["S_collab"]),
                reason=f"Restored via agent {aid}",
            )
        last_reason = (
            (s_errs[0] if s_errs else "")
            or trace.planner_failure_reason
            or (phys["reasons"][0] if phys["reasons"] else "")
            or "goal V not reached"
        )
    return ProbeTrial(
        mode=relax_mode,
        success=False,
        planner_feasible=False,
        physical_passed=False,
        goal_reached=False,
        joint_success=False,
        reason=last_reason,
    )


def execute_round_in_process(
    spec_path: Path,
    gen_dir: Path,
    eval_dir: Path,
    run_probes: bool = True,
    require_sapien: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": False,
        "runtime_error": None,
        "traceback": "",
        "schema_errors": [],
        "simulator_engine": "unknown",
        "physical": {},
        "milestones": {},
        "probe_report": {},
        "team_trace_stats": {},
        "keyframes": [],
        "rendered_frames": [],
    }
    try:
        spec = load_task_spec(spec_path)
        if str(gen_dir) not in sys.path:
            sys.path.insert(0, str(gen_dir))
        if str(eval_dir) not in sys.path:
            sys.path.insert(0, str(eval_dir))

        expert_file = gen_dir / "expert.py"
        eval_file = eval_dir / "evaluator.py"
        if not expert_file.exists():
            raise FileNotFoundError(f"Missing generator deliverable: {expert_file}")
        if not eval_file.exists():
            raise FileNotFoundError(f"Missing evaluator deliverable: {eval_file}")

        expert_mod = _load_module("harness_candidate_expert", expert_file)
        eval_mod = _load_module("harness_candidate_evaluator", eval_file)
        if not hasattr(expert_mod, "rollout_expert"):
            raise AttributeError("gen/expert.py must define `rollout_expert(spec, mode) -> EpisodeTrace`")
        if not hasattr(eval_mod, "evaluate_task"):
            raise AttributeError("eval/evaluator.py must define `evaluate_task(spec, trace) -> dict`")

        phys_gate = PhysicalGate(spec)

        # 1. Nominal team rollout (with rendering enabled)
        team_mode = ExecutionMode(mode="team", seed=0)
        team_trace, team_phys, team_miles, schema_errors = _eval_one_mode(
            spec, expert_mod, eval_mod, phys_gate, team_mode, require_sapien=require_sapien
        )
        out["schema_errors"] = schema_errors
        out["simulator_engine"] = str(team_trace.metadata.get("simulator_engine", "analytical"))
        out["physical"] = team_phys
        out["milestones"] = team_miles

        sample_contacts = sorted(
            {(c.entity_a, c.entity_b, bool(c.whitelisted_grasp)) for fr in team_trace.frames for c in fr.contacts}
        )[:24]
        sample_regions = {
            bn: sorted({r for fr in team_trace.frames if bn in fr.bodies for r in fr.bodies[bn].in_regions})
            for bn in sorted(team_trace.initial_bodies.keys())
        }
        out["team_trace_stats"] = {
            "simulator_engine": out["simulator_engine"],
            "planner_feasible": team_trace.planner_feasible,
            "planner_failure_reason": team_trace.planner_failure_reason,
            "n_frames": len(team_trace.frames),
            "duration_s": round(team_trace.duration_s, 3),
            "dt_s": team_trace.dt_s,
            "bodies": sorted(team_trace.initial_bodies.keys()),
            "arm_keys": sorted(team_trace.frames[0].arms.keys()) if team_trace.frames else [],
            "observed_contacts_sample": [list(x) for x in sample_contacts],
            "observed_regions": sample_regions,
        }
        out["keyframes"] = team_trace.sample_keyframes(max_frames=100)
        rendered = list(team_trace.metadata.get("rendered_frames", []))
        if len(rendered) > 28:
            step = max(1, len(rendered) // 26)
            picked = rendered[::step]
            if picked[-1]["step"] != rendered[-1]["step"]:
                picked.append(rendered[-1])
            rendered = picked
        out["rendered_frames"] = rendered

        # 2. Appendix A operational probes (disable RGB rendering during probe sweep for speed via env)
        if run_probes:
            os.environ["MAWAM_HARNESS_DISABLE_RENDER"] = "1"
            try:
                trials: List[ProbeTrial] = []
                team_ok = bool(
                    team_trace.planner_feasible
                    and team_phys["passed"]
                    and team_miles["joint_success"]
                    and not schema_errors
                )
                trials.append(
                    ProbeTrial(
                        mode="team",
                        success=team_ok,
                        planner_feasible=team_trace.planner_feasible,
                        physical_passed=bool(team_phys["passed"]),
                        goal_reached=bool(team_miles["goal_reached"]),
                        joint_success=bool(team_miles["joint_success"]),
                        S_role=float(team_miles["S_role"]),
                        S_collab=float(team_miles["S_collab"]),
                        reason="" if team_ok else (
                            (schema_errors[0] if schema_errors else "")
                            or (team_phys["reasons"][0] if team_phys["reasons"] else "")
                            or (team_miles["contract_errors"][0] if team_miles["contract_errors"] else "")
                            or "milestones incomplete"
                        ),
                    )
                )

                for aid in spec.agent_ids:
                    em = ExecutionMode(mode=f"single_{aid}", primary_agent=aid, seed=0)
                    tr, ph, mi, se = _eval_one_mode(spec, expert_mod, eval_mod, phys_gate, em, require_sapien=require_sapien)
                    ok = bool(tr.planner_feasible and ph["passed"] and mi["goal_reached"] and not se)
                    trials.append(
                        ProbeTrial(
                            mode=f"single_{aid}",
                            success=ok,
                            planner_feasible=tr.planner_feasible,
                            physical_passed=bool(ph["passed"]),
                            goal_reached=bool(mi["goal_reached"]),
                            joint_success=bool(mi["joint_success"]),
                            S_role=float(mi["S_role"]),
                            S_collab=float(mi["S_collab"]),
                            reason="" if ok else (tr.planner_failure_reason or (ph["reasons"][0] if ph["reasons"] else "goal V not reached")),
                        )
                    )

                for rmode in ("relax_O", "relax_W", "relax_U"):
                    trials.append(_run_relax_probe(spec, expert_mod, eval_mod, phys_gate, rmode, require_sapien=require_sapien))

                if spec.probes.serialize:
                    em = ExecutionMode(mode="serialize", seed=0)
                    tr, ph, mi, se = _eval_one_mode(spec, expert_mod, eval_mod, phys_gate, em, require_sapien=require_sapien)
                    ok = bool(tr.planner_feasible and ph["passed"] and mi["goal_reached"] and not se)
                    trials.append(
                        ProbeTrial(
                            mode="serialize",
                            success=ok,
                            planner_feasible=tr.planner_feasible,
                            physical_passed=bool(ph["passed"]),
                            goal_reached=bool(mi["goal_reached"]),
                            joint_success=bool(mi["joint_success"]),
                            S_role=float(mi["S_role"]),
                            S_collab=float(mi["S_collab"]),
                            reason="" if ok else (tr.planner_failure_reason or (ph["reasons"][0] if ph["reasons"] else "serialization broke goal V")),
                        )
                    )

                for d in spec.probes.delay_seconds:
                    em = ExecutionMode(mode="delay", delay_s=float(d), seed=0)
                    tr, ph, mi, se = _eval_one_mode(spec, expert_mod, eval_mod, phys_gate, em, require_sapien=require_sapien)
                    ok = bool(tr.planner_feasible and ph["passed"] and mi["joint_success"] and not se)
                    trials.append(
                        ProbeTrial(
                            mode=f"delay:{d:g}s",
                            success=ok,
                            planner_feasible=tr.planner_feasible,
                            physical_passed=bool(ph["passed"]),
                            goal_reached=bool(mi["goal_reached"]),
                            joint_success=bool(mi["joint_success"]),
                            S_role=float(mi["S_role"]),
                            S_collab=float(mi["S_collab"]),
                            reason="" if ok else (tr.planner_failure_reason or (ph["reasons"][0] if ph["reasons"] else "delay broke collaboration")),
                        )
                    )

                if spec.probes.role_swap or spec.labels.who == "heterogeneous":
                    em = ExecutionMode(mode="role_swap", seed=0)
                    tr, ph, mi, se = _eval_one_mode(spec, expert_mod, eval_mod, phys_gate, em, require_sapien=require_sapien)
                    ok = bool(tr.planner_feasible and ph["passed"] and mi["goal_reached"] and not se)
                    trials.append(
                        ProbeTrial(
                            mode="role_swap",
                            success=ok,
                            planner_feasible=tr.planner_feasible,
                            physical_passed=bool(ph["passed"]),
                            goal_reached=bool(mi["goal_reached"]),
                            joint_success=bool(mi["joint_success"]),
                            S_role=float(mi["S_role"]),
                            S_collab=float(mi["S_collab"]),
                            reason="" if ok else (tr.planner_failure_reason or (ph["reasons"][0] if ph["reasons"] else "role_swap infeasible")),
                        )
                    )

                out["probe_report"] = certify_probes(spec, trials).to_dict()
            finally:
                os.environ.pop("MAWAM_HARNESS_DISABLE_RENDER", None)

        out["ok"] = True
    except BaseException as exc:  # noqa: BLE001
        out["ok"] = False
        out["runtime_error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
    return out


def _pack_bundle(repo_harness_dir: Path, spec_path: Path, gen_dir: Path, eval_dir: Path) -> bytes:
    """Create a clean tar stream containing ma_task_harness/, TaskSpec.yaml, gen/, eval/."""
    with tempfile.SpooledTemporaryFile(max_size=20 * 1024 * 1024) as buf:
        with tarfile.open(fileobj=buf, mode="w") as tf:
            for p in sorted(repo_harness_dir.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith("._"):
                    tf.add(p, arcname=f"ma_task_harness/{p.relative_to(repo_harness_dir)}")
            tf.add(spec_path, arcname="work/TaskSpec.yaml")
            for sub, d in (("work/gen", gen_dir), ("work/eval", eval_dir)):
                if d.is_dir():
                    for p in sorted(d.rglob("*.py")):
                        if "__pycache__" not in p.parts and not p.name.startswith("._"):
                            tf.add(p, arcname=f"{sub}/{p.relative_to(d)}")
        buf.seek(0)
        return buf.read()


def run_candidate_subprocess(
    spec_path: Path,
    gen_dir: Path,
    eval_dir: Path,
    sim_python: Optional[str] = None,
    run_probes: bool = True,
    require_sapien: bool = False,
    sim_ssh: Optional[str] = None,
    sim_ssh_config: Optional[str] = None,
    sim_env_script: Optional[str] = None,
    sim_gpu: str = "0",
    timeout_s: int = 240,
) -> Dict[str, Any]:
    """Execute candidate `gen/` + `eval/` inside an isolated local or remote GPU subprocess."""
    marker = "__MAWAM_HARNESS_RUNNER_JSON__"
    repo_root = Path(__file__).resolve().parent.parent.parent
    harness_dir = Path(__file__).resolve().parent.parent

    if sim_ssh:
        tar_bytes = _pack_bundle(harness_dir, spec_path.resolve(), gen_dir.resolve(), eval_dir.resolve())
        py = sim_python or "python3"
        flags = []
        if not run_probes:
            flags.append("--no-probes")
        if require_sapien:
            flags.append("--require-sapien")
        env_prefix = f"source {shlex.quote(sim_env_script)} && " if sim_env_script else ""
        remote_cmd = (
            'TMPD=$(mktemp -d /tmp/mawam_sim_XXXXXX) && trap "rm -rf $TMPD" EXIT && '
            'tar -C "$TMPD" -xf - && '
            f'{env_prefix}'
            f'CUDA_VISIBLE_DEVICES={shlex.quote(str(sim_gpu))} '
            'PYTHONPATH="$TMPD:${PYTHONPATH:-}" '
            f'{shlex.quote(py)} -m ma_task_harness.runtime.subprocess_runner '
            '--spec "$TMPD/work/TaskSpec.yaml" '
            '--gen-dir "$TMPD/work/gen" '
            '--eval-dir "$TMPD/work/eval" '
            + " ".join(flags)
        )
        ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
        if sim_ssh_config:
            ssh_cmd.extend(["-F", sim_ssh_config])
        ssh_cmd.extend([sim_ssh, remote_cmd])
        try:
            proc = subprocess.run(ssh_cmd, input=tar_bytes, capture_output=True, timeout=timeout_s)
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return _timeout_payload(timeout_s)
    else:
        py = sim_python or sys.executable
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else str(repo_root)
        env.setdefault("CUDA_VISIBLE_DEVICES", str(sim_gpu))
        cmd = [
            py,
            "-m",
            "ma_task_harness.runtime.subprocess_runner",
            "--spec",
            str(spec_path.resolve()),
            "--gen-dir",
            str(gen_dir.resolve()),
            "--eval-dir",
            str(eval_dir.resolve()),
        ]
        if not run_probes:
            cmd.append("--no-probes")
        if require_sapien:
            cmd.append("--require-sapien")
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s, env=env, stdin=subprocess.DEVNULL
            )
            stdout, stderr = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            return _timeout_payload(timeout_s)

    if marker in stdout:
        pre, payload_str = stdout.split(marker, 1)
        data = json.loads(payload_str.strip())
        data["stdout"] = pre.strip()
        data["stderr"] = stderr.strip()
        return data

    return {
        "ok": False,
        "runtime_error": f"Subprocess exited with rc={proc.returncode} without JSON payload",
        "traceback": stderr[-2500:],
        "stdout": stdout[-1500:],
        "stderr": stderr[-2500:],
        "schema_errors": [],
        "simulator_engine": "unknown",
        "physical": {"passed": False, "reasons": [f"Subprocess crashed (rc={proc.returncode})"]},
        "milestones": {"joint_success": False, "S_role": 0.0, "S_collab": 0.0, "flags": {}},
        "probe_report": {},
        "keyframes": [],
        "rendered_frames": [],
    }


def _timeout_payload(timeout_s: int) -> Dict[str, Any]:
    return {
        "ok": False,
        "runtime_error": f"Candidate simulation timed out after {timeout_s}s",
        "traceback": "",
        "stdout": "",
        "stderr": "",
        "schema_errors": [],
        "simulator_engine": "unknown",
        "physical": {"passed": False, "reasons": [f"Subprocess timeout ({timeout_s}s)"]},
        "milestones": {"joint_success": False, "S_role": 0.0, "S_collab": 0.0, "flags": {}},
        "probe_report": {},
        "keyframes": [],
        "rendered_frames": [],
    }


def _cli() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--gen-dir", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--no-probes", action="store_true")
    p.add_argument("--require-sapien", action="store_true")
    args = p.parse_args()
    res = execute_round_in_process(
        args.spec,
        args.gen_dir,
        args.eval_dir,
        run_probes=not args.no_probes,
        require_sapien=args.require_sapien,
    )
    print("__MAWAM_HARNESS_RUNNER_JSON__")
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
