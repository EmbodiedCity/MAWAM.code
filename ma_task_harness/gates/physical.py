"""Task-agnostic LLM-free physical acceptance gates (Paper Appendix B.2 (i)–(iv)).

Applies identically to *any* multi-robot task without hardcoding object names:
  (i)   No robot–robot, robot–fixture, or self-collision outside whitelisted grasps
  (ii)  Inverse kinematics continuous (no joint jump > limit) and within joint bounds
  (iii) Real opposed finger–object contacts without pose overwriting / teleportation
  (iv)  All non-fixture dynamic bodies physically supported & at rest in terminal window
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ma_task_harness.runtime.state_types import EpisodeTrace
from ma_task_harness.spec.schema import TaskSpec


@dataclass
class PhysicalReport:
    collision_free: bool = True
    ik_continuous: bool = True
    grasps_persistent: bool = True
    terminal_at_rest: bool = True
    planner_feasible: bool = True
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (
            self.collision_free
            and self.ik_continuous
            and self.grasps_persistent
            and self.terminal_at_rest
            and self.planner_feasible
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "collision_free": self.collision_free,
            "ik_continuous": self.ik_continuous,
            "grasps_persistent": self.grasps_persistent,
            "terminal_at_rest": self.terminal_at_rest,
            "planner_feasible": self.planner_feasible,
            "reasons": list(self.reasons),
            "details": dict(self.details),
        }


class PhysicalGate:
    """Universal physical acceptance gate parameterized purely by `TaskSpec`."""

    def __init__(
        self,
        spec: TaskSpec,
        max_joint_jump_rad: float = 0.35,
        terminal_rest_s: Optional[float] = None,
        max_terminal_speed: Optional[float] = None,
        max_grasp_tilt_deg: float = 45.0,
    ) -> None:
        self.spec = spec
        self.agent_prefixes = tuple(f"{aid}." for aid in spec.agent_ids) + ("robot_",)
        self.max_joint_jump_rad = float(spec.thresholds.get("max_joint_jump_rad", max_joint_jump_rad))
        self.terminal_rest_s = float(
            terminal_rest_s
            if terminal_rest_s is not None
            else spec.thresholds.get("terminal_rest_s", 0.80)
        )
        # Allow any task-defined rest speed threshold, falling back to 0.05 m/s
        default_speed = (
            spec.thresholds.get("terminal_rest_speed_mps")
            or spec.thresholds.get("rest_speed_mps")
            or spec.thresholds.get("linear_rest_mps")
            or spec.thresholds.get("pot_rest_speed_mps")
            or 0.05
        )
        self.max_terminal_speed = float(max_terminal_speed if max_terminal_speed is not None else default_speed)
        self.max_grasp_tilt_deg = float(spec.thresholds.get("max_grasp_tilt_deg", max_grasp_tilt_deg))

    def _is_robot(self, entity: str) -> bool:
        return entity.startswith(self.agent_prefixes)

    def _is_fixture(self, entity: str, bodies: Dict[str, Any]) -> bool:
        b = bodies.get(entity)
        if b is not None and getattr(b, "is_fixture", False):
            return True
        return entity in ("table", "ground", "stand", "wall") or entity.endswith(
            ("_frame", "_rim", "_wall", "_shelf", "_fixture", "_cage", "_gate")
        )

    def evaluate(self, trace: EpisodeTrace) -> PhysicalReport:
        report = PhysicalReport(planner_feasible=bool(trace.planner_feasible))
        if not trace.planner_feasible:
            report.reasons.append(
                f"Planner infeasible under {trace.execution_mode.label()}: {trace.planner_failure_reason}"
            )
            return report
        if not trace.frames:
            report.planner_feasible = False
            report.reasons.append("Empty episode: zero control frames recorded.")
            return report

        self._check_collisions(trace, report)
        self._check_ik(trace, report)
        self._check_grasps(trace, report)
        self._check_terminal_rest(trace, report)
        return report

    def _check_collisions(self, trace: EpisodeTrace, report: PhysicalReport) -> None:
        bad: List[str] = []
        for fr in trace.frames:
            for c in fr.contacts:
                if c.whitelisted_grasp:
                    continue
                ra, rb = self._is_robot(c.entity_a), self._is_robot(c.entity_b)
                fa, fb = self._is_fixture(c.entity_a, fr.bodies), self._is_fixture(c.entity_b, fr.bodies)
                # Forbid non-whitelisted robot-robot, robot-fixture, and object-fixture rim impacts
                if (ra and rb) or (ra and fb) or (rb and fa):
                    bad.append(
                        f"t={fr.time_s:.3f}s collision {c.entity_a}<->{c.entity_b} (F={c.normal_force_n:.1f}N)"
                    )
        if bad:
            report.collision_free = False
            extra = f" (+{len(bad)-1} more)" if len(bad) > 1 else ""
            report.reasons.append(f"Non-whitelisted collision: {bad[0]}{extra}")
            report.details["collisions"] = bad[:15]

    def _check_ik(self, trace: EpisodeTrace, report: PhysicalReport) -> None:
        prev_q: Dict[str, Tuple[float, ...]] = {}
        issues: List[str] = []
        for fr in trace.frames:
            for key, arm in fr.arms.items():
                if not arm.ik_valid:
                    issues.append(f"t={fr.time_s:.3f}s {key}: IK solver invalid")
                    continue
                if arm.qpos and arm.joint_lower and arm.joint_upper:
                    for qi, (q, lo, hi) in enumerate(zip(arm.qpos, arm.joint_lower, arm.joint_upper)):
                        if q < lo - 1e-6 or q > hi + 1e-6:
                            issues.append(f"t={fr.time_s:.3f}s {key} q[{qi}]={q:.3f} outside [{lo:.3f}, {hi:.3f}]")
                if key in prev_q and arm.qpos and len(prev_q[key]) == len(arm.qpos):
                    jump = max(abs(a - b) for a, b in zip(arm.qpos, prev_q[key]))
                    if jump > self.max_joint_jump_rad + 1e-9:
                        issues.append(f"t={fr.time_s:.3f}s {key}: joint jump {jump:.3f} rad > {self.max_joint_jump_rad}")
                if arm.qpos:
                    prev_q[key] = arm.qpos
        if issues:
            report.ik_continuous = False
            report.reasons.append(issues[0])
            report.details["ik_issues"] = issues[:15]

    def _check_grasps(self, trace: EpisodeTrace, report: PhysicalReport) -> None:
        failures: List[str] = []
        for fr in trace.frames:
            if fr.pose_overwritten:
                failures.append(f"t={fr.time_s:.3f}s synthetic pose overwrite detected (forbidden)")
                continue
            grasped: Set[str] = set()
            for c in fr.contacts:
                if not c.whitelisted_grasp:
                    continue
                if self._is_robot(c.entity_a):
                    grasped.add(c.entity_b)
                if self._is_robot(c.entity_b):
                    grasped.add(c.entity_a)
            for bname in grasped:
                body = fr.bodies.get(bname)
                if body is None or body.is_fixture:
                    continue
                opposed = sum(
                    1
                    for c in fr.contacts
                    if c.whitelisted_grasp
                    and c.two_finger_opposed
                    and (c.entity_a == bname or c.entity_b == bname)
                )
                airborne = len(body.supported_by) == 0
                if airborne and opposed == 0:
                    failures.append(f"t={fr.time_s:.3f}s airborne {bname} has zero opposed finger contacts")
                if abs(body.tilt_deg) > self.max_grasp_tilt_deg and opposed > 0:
                    failures.append(
                        f"t={fr.time_s:.3f}s {bname} tilt={body.tilt_deg:.1f}deg > {self.max_grasp_tilt_deg}deg during grasp (slip)"
                    )
        if failures:
            report.grasps_persistent = False
            report.reasons.append(failures[0])
            report.details["grasp_failures"] = failures[:15]

    def _check_terminal_rest(self, trace: EpisodeTrace, report: PhysicalReport) -> None:
        n_tail = max(1, int(round(self.terminal_rest_s / max(trace.dt_s, 1e-6))))
        tail = trace.frames[-n_tail:]
        moving: List[str] = []
        max_spd = 0.0
        for fr in tail:
            for bname, body in fr.bodies.items():
                if body.is_fixture:
                    continue
                spd = body.linear_speed
                max_spd = max(max_spd, spd)
                if spd > self.max_terminal_speed + 1e-9:
                    moving.append(
                        f"t={fr.time_s:.3f}s {bname} speed={spd*100:.2f}cm/s > {self.max_terminal_speed*100:.2f}cm/s"
                    )
        if moving:
            report.terminal_at_rest = False
            report.reasons.append(moving[0])
            report.details["terminal_motion"] = moving[:15]
            report.details["max_terminal_speed_mps"] = round(max_spd, 5)
