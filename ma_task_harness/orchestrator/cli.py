"""Task-agnostic CLI for MA-Task Harness (`python3 -m ma_task_harness`).

Commands:
  specify       Stage 1: Natural-language idea -> multi-round Why/When/Who clarification -> TaskSpec.yaml
  validate      Validate ANY TaskSpec.yaml (or all MA-Bench examples)
  coevolve      Stage 2 & 3: Dual-agent G^(n) <-> E^(n) code co-evolution + probes + package freeze
  verify-pkg    Re-verify ANY frozen task package (gen/ + eval/ + TaskSpec.yaml)
  render-trace  Generate standalone interactive HTML trace viewer from `harness_trace.json`
  list-examples List the 7 MA-Bench example TaskSpecs included in `examples/mabench/`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from ma_task_harness.orchestrator.backends import make_backend
from ma_task_harness.orchestrator.driver import CoEvolutionDriver
from ma_task_harness.orchestrator.specifier import run_specify
from ma_task_harness.runtime.subprocess_runner import run_candidate_subprocess
from ma_task_harness.spec.schema import examples_dir, load_example_specs, load_task_spec
from ma_task_harness.viz.render_trace import render_trace_html


def _resolve_spec_path(spec_arg: str) -> Path:
    p = Path(spec_arg)
    if p.is_file():
        return p.resolve()
    # Allow referencing an example task by slug or short alias
    ex_dir = examples_dir()
    aliases = {
        "bowl": "joint_bowl_placing",
        "color": "find_common_color",
        "scan": "hold_and_scan",
        "door": "open_door_place_pot",
        "tray": "lift_tray_keep_blocks",
        "food": "lift_tray_pour_food",
        "balls": "pour_and_catch",
    }
    slug = aliases.get(spec_arg, spec_arg)
    cand = ex_dir / f"{slug}.yaml"
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(
        f"TaskSpec not found at {spec_arg!r}. Pass a path to any custom TaskSpec.yaml "
        f"or an example slug from `list-examples`."
    )


def _cmd_list_examples(_: argparse.Namespace) -> int:
    specs = load_example_specs()
    print(f"{'example_slug':28} {'why':12} {'when':12} {'who':14} title")
    print("-" * 92)
    for slug, sp in specs.items():
        print(f"{slug:28} {sp.labels.why:12} {sp.labels.when:12} {sp.labels.who:14} {sp.title}")
    print(f"\n{len(specs)} MA-Bench example TaskSpecs in {examples_dir()}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    if args.spec:
        sp_path = _resolve_spec_path(args.spec)
        sp = load_task_spec(sp_path)
        print(f"OK  {sp.task}  agents={sp.agent_ids}  labels={sp.labels.to_dict()}  sha256={sp.sha256()[:16]}...")
        return 0
    specs = load_example_specs()
    for sp in specs.values():
        print(f"OK  {sp.task:28} agents={sp.agent_ids}  sha256={sp.sha256()[:16]}...")
    print(f"Validated {len(specs)} example TaskSpecs.")
    return 0


def _cmd_specify(args: argparse.Namespace) -> int:
    backend = make_backend(args.backend, model=args.model)
    answers: List[str] = list(args.answer or [])
    if args.answers_json:
        answers.extend(json.loads(Path(args.answers_json).read_text(encoding="utf-8")))
    spec, trace = run_specify(
        idea=args.idea,
        backend=backend,
        work_root=Path(args.work_dir),
        clarification_answers=answers,
        interactive=bool(args.interactive),
        max_clarify_rounds=args.clarify_rounds,
    )
    print(
        json.dumps(
            {
                "status": "specified",
                "task": spec.task,
                "title": spec.title,
                "labels": spec.labels.to_dict(),
                "team": {k: v.robot for k, v in spec.team.items()},
                "task_spec_path": trace["task_spec_path"],
            },
            indent=2,
        )
    )
    return 0


def _cmd_coevolve(args: argparse.Namespace) -> int:
    sp_path = _resolve_spec_path(args.spec)
    sp = load_task_spec(sp_path)
    backend = make_backend(args.backend, model=args.model)
    work = Path(args.work_dir) if args.work_dir else Path.cwd() / "harness_runs" / sp.task
    driver = CoEvolutionDriver(
        spec=sp,
        backend=backend,
        work_root=work,
        max_rounds=args.max_rounds,
        sim_backend=args.sim_backend,
        sim_python=args.sim_python,
        sim_ssh=args.sim_ssh,
        sim_ssh_config=args.sim_ssh_config,
        sim_env_script=args.sim_env_script,
        sim_gpu=args.sim_gpu,
    )
    result = driver.run_coevolution(
        initial_gen_dir=Path(args.init_gen) if args.init_gen else None,
        initial_eval_dir=Path(args.init_eval) if args.init_eval else None,
        resume=bool(args.resume),
    )
    print(
        json.dumps(
            {
                "task": result.task,
                "converged": result.converged,
                "rounds": len(result.rounds),
                "sim_backend": args.sim_backend,
                "probe_matches_labels": result.probe_report.get("matches_labels"),
                "certified": {
                    "why": result.probe_report.get("why_certified"),
                    "when": result.probe_report.get("when_certified"),
                    "who": result.probe_report.get("who_certified"),
                },
                "package_dir": result.package_dir,
                "trace_path": result.trace_path,
            },
            indent=2,
        )
    )
    if args.render_html and result.trace_path:
        out_html = Path(args.render_html)
        render_trace_html(Path(result.trace_path), out_html)
        print(f"Rendered interactive HTML trace -> {out_html}")
    return 0 if result.converged else 1


def _cmd_verify_pkg(args: argparse.Namespace) -> int:
    pkg = Path(args.package_dir).resolve()
    res = run_candidate_subprocess(
        spec_path=pkg / "TaskSpec.yaml",
        gen_dir=pkg / "gen",
        eval_dir=pkg / "eval",
        sim_python=args.sim_python,
        run_probes=True,
        require_sapien=(args.sim_backend == "sapien"),
        sim_ssh=args.sim_ssh,
        sim_ssh_config=args.sim_ssh_config,
        sim_env_script=args.sim_env_script,
        sim_gpu=args.sim_gpu,
    )
    ok = bool(
        res["ok"]
        and not res["schema_errors"]
        and res["physical"]["passed"]
        and res["milestones"]["joint_success"]
        and res["probe_report"]["matches_labels"]
    )
    print(
        json.dumps(
            {
                "package": str(pkg),
                "ok": ok,
                "simulator_engine": res.get("simulator_engine"),
                "rendered_frames": len(res.get("rendered_frames", [])),
                "physical": res["physical"],
                "milestones": {
                    "S_role": res["milestones"].get("S_role"),
                    "S_collab": res["milestones"].get("S_collab"),
                    "joint_success": res["milestones"].get("joint_success"),
                    "flags": res["milestones"].get("flags"),
                },
                "probes": {
                    "matches_labels": res["probe_report"].get("matches_labels"),
                    "why": res["probe_report"].get("why_certified"),
                    "when": res["probe_report"].get("when_certified"),
                    "who": res["probe_report"].get("who_certified"),
                },
            },
            indent=2,
        )
    )
    return 0 if ok else 1


def _cmd_render_trace(args: argparse.Namespace) -> int:
    src = Path(args.trace)
    if src.is_dir():
        src = src / "harness_trace.json"
    out = Path(args.out) if args.out else src.with_name("trace_viewer.html")
    render_trace_html(src, out)
    print(f"Rendered HTML trace viewer -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mawam-harness",
        description="MA-Task Harness: task-agnostic coding-agent co-evolution for multi-robot tasks (MAWAM §3.2).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("list-examples", help="List the 7 MA-Bench example TaskSpecs")
    s.set_defaults(func=_cmd_list_examples)

    s = sub.add_parser("validate", help="Validate any custom TaskSpec.yaml (or all examples)")
    s.add_argument("--spec", default=None, help="Path to any TaskSpec.yaml or example slug")
    s.set_defaults(func=_cmd_validate)

    s = sub.add_parser("specify", help="Stage 1: Clarify any task idea into a validated TaskSpec.yaml")
    s.add_argument("--idea", required=True, help="Natural-language description of a custom multi-robot task")
    s.add_argument("--backend", default="claude", choices=("claude", "codex", "openai"))
    s.add_argument("--model", default=None)
    s.add_argument("--work-dir", required=True)
    s.add_argument("--interactive", action="store_true")
    s.add_argument("--clarify-rounds", type=int, default=2)
    s.add_argument("--answer", action="append", default=None, help="Clarification answer (repeatable)")
    s.add_argument("--answers-json", default=None, help="JSON list of clarification answers")
    s.set_defaults(func=_cmd_specify)

    s = sub.add_parser("coevolve", help="Stage 2 & 3: Co-evolve gen/ and eval/ for ANY TaskSpec.yaml")
    s.add_argument("--spec", required=True, help="Path to any custom TaskSpec.yaml (or example slug)")
    s.add_argument("--backend", default="claude", choices=("claude", "codex", "openai"))
    s.add_argument("--model", default=None)
    s.add_argument("--work-dir", default=None)
    s.add_argument("--max-rounds", type=int, default=5)
    s.add_argument("--sim-backend", default="sapien", choices=("sapien", "any"), help="Require real SAPIEN+PhysX ('sapien') or allow CPU mode ('any')")
    s.add_argument("--sim-python", default=None, help="Python interpreter with sapien + mplib installed")
    s.add_argument("--sim-ssh", default=None, help="Optional remote GPU SSH host for SAPIEN execution")
    s.add_argument("--sim-ssh-config", default=None, help="Optional SSH config path for --sim-ssh")
    s.add_argument("--sim-env-script", default=None, help="Optional remote env script to source before running SAPIEN")
    s.add_argument("--sim-gpu", default="0", help="CUDA_VISIBLE_DEVICES index for SAPIEN rendering")
    s.add_argument("--init-gen", default=None, help="Optional seed gen/ directory")
    s.add_argument("--init-eval", default=None, help="Optional seed eval/ directory")
    s.add_argument("--resume", action="store_true", help="Resume from existing harness_trace.json in work-dir")
    s.add_argument("--render-html", default=None, help="Optional path to render trace_viewer.html upon completion")
    s.set_defaults(func=_cmd_coevolve)

    s = sub.add_parser("verify-pkg", help="Re-verify any frozen task package (TaskSpec + gen/ + eval/)")
    s.add_argument("package_dir", help="Path to frozen package/ directory")
    s.add_argument("--sim-backend", default="sapien", choices=("sapien", "any"))
    s.add_argument("--sim-python", default=None)
    s.add_argument("--sim-ssh", default=None)
    s.add_argument("--sim-ssh-config", default=None)
    s.add_argument("--sim-env-script", default=None)
    s.add_argument("--sim-gpu", default="0")
    s.set_defaults(func=_cmd_verify_pkg)

    s = sub.add_parser("render-trace", help="Render harness_trace.json as a standalone interactive HTML report")
    s.add_argument("trace", help="Path to harness_trace.json or run directory")
    s.add_argument("--out", default=None, help="Output HTML path")
    s.set_defaults(func=_cmd_render_trace)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
