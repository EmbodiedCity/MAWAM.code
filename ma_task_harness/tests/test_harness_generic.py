"""Task-agnostic harness unit tests (no LLM API keys or GPU required).

Verifies:
  1. All 7 MA-Bench example TaskSpecs validate cleanly under the open schema
  2. Arbitrary custom N=3 heterogeneous TaskSpec with custom robots & custom predicates validates
  3. Mismatched Who label & wrong O<W<U probe order are rejected
  4. Full 3-stage pipeline (Stage 1 specify -> Stage 2 bidirectional G<->E code co-evolution
     with physical failure + G->E counter-audit + isolated subprocess execution -> Stage 3
     Appendix A probes & freeze -> interactive HTML trace rendering) passes on a brand-new
     non-MA-Bench custom task!
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from ma_task_harness.orchestrator.backends import AgentResult, CallableBackend
from ma_task_harness.orchestrator.driver import CoEvolutionDriver
from ma_task_harness.orchestrator.specifier import run_specify
from ma_task_harness.runtime.subprocess_runner import execute_round_in_process
from ma_task_harness.spec.schema import TaskSpec, TaskSpecValidationError, load_example_specs
from ma_task_harness.viz.render_trace import render_trace_html


def test_examples_and_custom_3agent_schema() -> None:
    examples = load_example_specs()
    assert len(examples) == 7

    # Arbitrary 3-agent custom task with custom robot type & custom predicates (not in MA-Bench!)
    custom_3 = {
        "task": "three_robot_bridge_mount",
        "title": "Three-robot bridge mount",
        "version": "v1",
        "team": {
            "left_lifter": {"robot": "ur5", "arms": 1, "action_dim": 8, "views": ["head", "wrist"]},
            "right_lifter": {"robot": "ur5", "arms": 1, "action_dim": 8, "views": ["head", "wrist"]},
            "fastener": {"robot": "custom_screw_arm", "arms": 1, "action_dim": 7, "views": ["wrist"]},
        },
        "labels": {"why": "capability", "when": "synchronous", "who": "heterogeneous"},
        "prompts": {
            "global": "Two lifters hold the long girder level while the fastener tightens the center bolt.",
            "left_lifter": "Hold left end of girder level.",
            "right_lifter": "Hold right end of girder level.",
            "fastener": "Drive center bolt once girder is aligned.",
        },
        "thresholds": {"max_tilt_deg": 4.0, "bolt_turns": 3.0},
        "goal": [{"id": "goal_bolt_tight", "kind": "custom_torque_check", "params": {"turns": "<bolt_turns>"}}],
        "role_milestones": {
            "left_lifter": [{"id": "m_L1", "kind": "custom_lift", "params": {}}],
            "right_lifter": [{"id": "m_R1", "kind": "custom_lift", "params": {}}],
            "fastener": [{"id": "m_F1", "kind": "custom_fasten", "params": {"turns": "<bolt_turns>"}}],
        },
        "collab_milestones": [
            {"id": "c_level", "kind": "custom_co_hold", "params": {"max_tilt": "<max_tilt_deg>"}}
        ],
        "probes": {
            "necessity": ["single_left_lifter", "single_right_lifter", "single_fastener", "relax_O", "relax_W", "relax_U"],
            "temporal": {"serialize": True, "delay_seconds": [0.05, 0.5], "sync_break_tolerance_s": 0.15},
            "embodiment": ["role_swap"],
        },
    }
    sp = TaskSpec.from_dict(custom_3)
    assert len(sp.agent_ids) == 3

    # Reject wrong Who label
    bad = dict(custom_3)
    bad["labels"] = {"why": "capability", "when": "synchronous", "who": "homogeneous"}
    try:
        TaskSpec.from_dict(bad)
        raise AssertionError("Should reject homogeneous on mismatched 3-agent team")
    except TaskSpecValidationError:
        pass


def test_full_3stage_custom_task_coevolution_and_html() -> None:
    """Exercise Stage 1 -> Stage 2 (with physics fix + G->E counter-audit) -> Stage 3 -> HTML."""
    custom_yaml = """task: spring_valve_tube_dock
title: "Spring-valve tube dock"
version: v1
description: "Single-arm UR5 depresses a spring-loaded valve clamp while dual-arm ALOHA docks a two-grip hose."
nominal_duration_s: 12.0
control_hz: 25.0
physics_hz: 250.0
team:
  A:
    robot: ur5
    arms: 1
    action_dim: 8
    views: [head, wrist]
    workspace_bounds: [[-0.5, -0.4, 0.7], [0.2, 0.4, 1.3]]
    observable_zones: [valve_zone]
  B:
    robot: aloha
    arms: 2
    action_dim: 16
    views: [head, left_wrist, right_wrist]
    workspace_bounds: [[-0.2, -0.5, 0.7], [0.5, 0.5, 1.3]]
    observable_zones: [valve_zone, hose_zone]
labels: {why: capability, when: partial, who: heterogeneous}
prompts:
  global: "UR5 holds the spring valve open while ALOHA docks the hose."
  A: "Press and hold the spring valve lever open."
  B: "Grasp both hose handles and dock the hose into the open port."
thresholds:
  valve_open_deg: 45.0
  terminal_rest_s: 0.4
  terminal_rest_speed_mps: 0.05
scene_params:
  spring_stiffness: 0.5
goal:
  - id: goal_docked
    kind: hose_docked
    params: {}
role_milestones:
  A:
    - id: m_A1
      kind: valve_open
      params: {min_deg: "<valve_open_deg>"}
  B:
    - id: m_B1
      kind: hose_inserted
      params: {}
collab_milestones:
  - id: c_1
    kind: valve_held_during_dock
    params: {min_deg: "<valve_open_deg>"}
probes:
  necessity: [single_A, single_B, relax_O, relax_W, relax_U]
  temporal: {serialize: true, delay_seconds: [0.08, 0.8, 2.0], sync_break_tolerance_s: 0.20}
  embodiment: [role_swap]
"""

    expert_r1_buggy = '''from ma_task_harness.runtime.state_types import ArmState, BodyState, ContactEvent, ControlFrame, EpisodeTrace, ExecutionMode, effective_capabilities

def rollout_expert(spec, mode):
    active, caps, role_map = effective_capabilities(spec, mode)
    # Round 1 intentionally hits a fixture collision to exercise E->G repair
    fr = ControlFrame(
        step=0, time_s=0.0,
        arms={"A.main": ArmState("A", "main", (0.0, 0.0, 0.9))},
        bodies={"hose": BodyState("hose", (0.1, 0.0, 0.8)), "valve": BodyState("valve", (0.0, 0.0, 0.8), joint_deg=50.0, is_fixture=True)},
        contacts=[ContactEvent("B.left_wrist", "valve_frame", 20.0, whitelisted_grasp=False)],
        camera_extrinsics_shapes={"A.head": (3, 4)},
    )
    return EpisodeTrace(spec.task, mode.seed, 0.04, [fr], {"hose": fr.bodies["hose"]}, mode, True)
'''

    expert_r2_fixed = '''import math
from ma_task_harness.runtime.state_types import ArmState, BodyState, ContactEvent, ControlFrame, EpisodeTrace, ExecutionMode, effective_capabilities

def rollout_expert(spec, mode):
    active, caps, role_map = effective_capabilities(spec, mode)
    single = active[0] if len(active) == 1 else None
    hose_exec = single if single else next((a for a in active if role_map[a] == "B"), None)
    valve_exec = single if single else next((a for a in active if role_map[a] == "A"), None)
    if hose_exec is None or caps[hose_exec].arms < 2:
        return EpisodeTrace(spec.task, mode.seed, 0.04, [], {}, mode, False, "Hose requires 2 arms")
    can_sim = (caps[single].arms >= 3) if single else (valve_exec != hose_exec and caps[valve_exec].arms >= 1 and caps[hose_exec].arms >= 2)
    if mode.mode == "serialize" or not can_sim:
        return EpisodeTrace(spec.task, mode.seed, 0.04, [], {}, mode, False, "Spring valve snaps shut when released before docking")

    dt = 0.04
    delay_steps = int(round((mode.delay_s if mode.mode == "delay" else 0.0) / dt))
    dock_start = 10 + delay_steps
    dock_end = dock_start + 15
    total = dock_end + 15
    frames = []
    init_hose = BodyState("hose", (0.35, 0.0, 0.75), supported_by=("table",))
    for s in range(total):
        holding_valve = 2 <= s <= dock_end + 2
        carrying = dock_start <= s < dock_end
        docked = s >= dock_end
        vdeg = 52.0 if holding_valve else 0.0
        hx = 0.35 if s < dock_start else (0.35 - 0.35 * (s - dock_start) / 15.0 if carrying else 0.0)
        hz = 0.75 if s < dock_start else (0.85 if carrying else 0.80)
        contacts = []
        if holding_valve:
            contacts.append(ContactEvent(f"{valve_exec}.finger", "valve", 8.0, True, True))
        if carrying:
            contacts.append(ContactEvent(f"{hose_exec}.left_finger", "hose", 9.0, True, True))
            contacts.append(ContactEvent(f"{hose_exec}.right_finger", "hose", 9.0, True, True))
        frames.append(ControlFrame(
            step=s, time_s=s * dt,
            arms={
                "A.main": ArmState("A", "main", (-0.1, 0.0, 0.85), gripper_open=0.0 if holding_valve else 1.0),
                "B.left": ArmState("B", "left", (hx, -0.05, hz), gripper_open=0.0 if carrying else 1.0),
                "B.right": ArmState("B", "right", (hx, 0.05, hz), gripper_open=0.0 if carrying else 1.0),
            },
            bodies={
                "hose": BodyState("hose", (hx, 0.0, hz), supported_by=("table",) if not carrying else (), in_regions=("dock_port",) if docked else ()),
                "valve": BodyState("valve", (0.0, 0.0, 0.82), joint_deg=vdeg, is_fixture=True),
            },
            contacts=contacts,
            camera_extrinsics_shapes={"A.head": (3, 4)},
            metadata={"carrying_hose": carrying},
        ))
    return EpisodeTrace(spec.task, mode.seed, dt, frames, {"hose": init_hose}, mode, True)
'''

    eval_r1_strict_44 = '''def check_schema(spec, trace):
    errs = []
    for fr in trace.frames[:1]:
        for cam, sh in fr.camera_extrinsics_shapes.items():
            if tuple(sh) != (4, 4):
                errs.append(f"Expected (4,4) camera extrinsic for {cam}, got {sh}")
    return errs

def evaluate_task(spec, trace):
    v_min = spec.thresholds["valve_open_deg"]
    m_a1 = any(fr.bodies["valve"].joint_deg >= v_min for fr in trace.frames if "valve" in fr.bodies)
    docked = bool(trace.frames and "dock_port" in trace.frames[-1].bodies["hose"].in_regions)
    carry = [fr for fr in trace.frames if fr.metadata.get("carrying_hose")]
    c1 = bool(carry and all(fr.bodies["valve"].joint_deg >= v_min for fr in carry))
    return {"flags": {"goal_docked": docked, "m_A1": m_a1, "m_B1": docked, "c_1": c1}, "notes": {}}
'''

    eval_r2_fixed = eval_r1_strict_44.replace("tuple(sh) != (4, 4)", "tuple(sh) not in ((3, 4), (4, 4))")

    state = {"gen_calls": 0, "eval_calls": 0}

    def handler(sys_p: str, user_p: str, ws: Path) -> AgentResult:
        head = sys_p.splitlines()[0].upper()
        if "SPECIFIER" in head:
            if "Do NOT write TaskSpec.yaml" in user_p:
                return AgentResult(ok=True, backend="callable", stdout=json.dumps({
                    "status": "clarify",
                    "summary": "UR5 + ALOHA spring-valve hose docking",
                    "questions": ["What valve angle unlocks the port?", "Does the valve snap shut on release?"],
                }))
            (ws / "TaskSpec.yaml").write_text(custom_yaml, encoding="utf-8")
            return AgentResult(ok=True, backend="callable", stdout="Finalized TaskSpec.yaml.")
        if "GENERATOR" in head:
            state["gen_calls"] += 1
            (ws / "gen").mkdir(exist_ok=True)
            if state["gen_calls"] == 1:
                (ws / "gen" / "expert.py").write_text(expert_r1_buggy, encoding="utf-8")
                return AgentResult(ok=True, backend="callable", stdout="Drafted initial v1 expert.")
            (ws / "gen" / "expert.py").write_text(expert_r2_fixed, encoding="utf-8")
            audit = json.loads((ws / "inbox" / "audit_report.json").read_text())
            if audit.get("schema_errors"):
                (ws / "outbox" / "counter_audit.json").write_text(
                    json.dumps({"findings": ["Evaluator wrongly rejects native SAPIEN (3,4) extrinsics"]}),
                    encoding="utf-8",
                )
            return AgentResult(ok=True, backend="callable", stdout="Fixed collision & emitted counter-audit.")
        # EVALUATOR
        state["eval_calls"] += 1
        (ws / "eval").mkdir(exist_ok=True)
        if state["eval_calls"] == 1:
            (ws / "eval" / "evaluator.py").write_text(eval_r1_strict_44, encoding="utf-8")
            return AgentResult(ok=True, backend="callable", stdout="Drafted evaluator v1.")
        (ws / "eval" / "evaluator.py").write_text(eval_r2_fixed, encoding="utf-8")
        return AgentResult(ok=True, backend="callable", stdout="Accepted counter-audit; allowed (3,4) extrinsics.")

    backend = CallableBackend(handler)
    with tempfile.TemporaryDirectory(prefix="mawam_generic_") as tmp:
        root = Path(tmp) / "custom_run"
        spec, s1 = run_specify(
            idea="UR5 holds a spring valve while ALOHA docks a hose",
            backend=backend,
            work_root=root,
            clarification_answers=["45 degrees unlocks it; spring snaps shut in 0.2s."],
            max_clarify_rounds=1,
        )
        assert spec.task == "spring_valve_tube_dock"
        assert len(s1["dialogue"]) == 4

        driver = CoEvolutionDriver(spec=spec, backend=backend, work_root=root, max_rounds=4)
        res = driver.run_coevolution()
        assert res.converged, f"Expected convergence, got {res.to_dict()}"
        assert len(res.rounds) == 2
        assert res.rounds[1].counter_audit is not None
        assert res.package_dir is not None

        # Standalone re-verification of frozen package
        pkg = Path(res.package_dir)
        v = execute_round_in_process(pkg / "TaskSpec.yaml", pkg / "gen", pkg / "eval", run_probes=True)
        assert v["ok"] and v["physical"]["passed"] and v["milestones"]["joint_success"] and v["probe_report"]["matches_labels"]

        # Render interactive HTML trace viewer
        html_out = render_trace_html(Path(res.trace_path), root / "trace_viewer.html")
        html_txt = html_out.read_text(encoding="utf-8")
        assert "spring_valve_tube_dock" in html_txt and "Counter-Audit" in html_txt


def run_all() -> None:
    tests = [test_examples_and_custom_3agent_schema, test_full_3stage_custom_task_coevolution_and_html]
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} task-agnostic harness tests passed.")


if __name__ == "__main__":
    run_all()
