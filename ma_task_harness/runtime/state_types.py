"""Task-agnostic multi-robot simulator state types & N-agent capability relaxations.

Shared contract between any generator (`gen/expert.py`), any deterministic evaluator
(`eval/evaluator.py`), universal physical gates (`gates/physical.py`), RoboTwin/SAPIEN
bridges, and the HTML trace visualizer (`viz/render_trace.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ma_task_harness.spec.schema import EmbodimentSpec, TaskSpec


@dataclass
class ArmState:
    """Per-arm kinematic and gripper state at one 25 Hz control step."""

    agent_id: str
    arm_name: str
    ee_pos: Tuple[float, float, float]
    ee_quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    gripper_open: float = 1.0  # 0.0 closed .. 1.0 open
    qpos: Tuple[float, ...] = ()
    joint_lower: Tuple[float, ...] = ()
    joint_upper: Tuple[float, ...] = ()
    ik_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "arm_name": self.arm_name,
            "ee_pos": list(self.ee_pos),
            "ee_quat_wxyz": list(self.ee_quat_wxyz),
            "gripper_open": float(self.gripper_open),
            "qpos": list(self.qpos),
            "joint_lower": list(self.joint_lower),
            "joint_upper": list(self.joint_upper),
            "ik_valid": bool(self.ik_valid),
        }


@dataclass
class BodyState:
    """Rigid or articulated object/fixture state at one control step."""

    name: str
    pos: Tuple[float, float, float]
    lin_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    tilt_deg: float = 0.0
    joint_deg: float = 0.0
    supported_by: Tuple[str, ...] = ("table",)
    in_regions: Tuple[str, ...] = ()
    container: Optional[str] = None
    is_fixture: bool = False  # True for hinged doors, levers, tables, cages, stands
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def linear_speed(self) -> float:
        vx, vy, vz = self.lin_vel
        return float((vx * vx + vy * vy + vz * vz) ** 0.5)

    @property
    def angular_speed(self) -> float:
        wx, wy, wz = self.ang_vel
        return float((wx * wx + wy * wy + wz * wz) ** 0.5)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pos": list(self.pos),
            "lin_vel": list(self.lin_vel),
            "ang_vel": list(self.ang_vel),
            "tilt_deg": float(self.tilt_deg),
            "joint_deg": float(self.joint_deg),
            "supported_by": list(self.supported_by),
            "in_regions": list(self.in_regions),
            "container": self.container,
            "is_fixture": bool(self.is_fixture),
            "attributes": dict(self.attributes),
        }


@dataclass
class ContactEvent:
    """Physical contact pair observed during the 250 Hz substeps of a control step."""

    entity_a: str
    entity_b: str
    normal_force_n: float = 1.0
    two_finger_opposed: bool = False
    whitelisted_grasp: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_a": self.entity_a,
            "entity_b": self.entity_b,
            "normal_force_n": float(self.normal_force_n),
            "two_finger_opposed": bool(self.two_finger_opposed),
            "whitelisted_grasp": bool(self.whitelisted_grasp),
        }


@dataclass
class ControlFrame:
    """One control step summarizing multi-robot poses, object states, and contacts."""

    step: int
    time_s: float
    arms: Dict[str, ArmState]
    bodies: Dict[str, BodyState]
    contacts: List[ContactEvent] = field(default_factory=list)
    camera_extrinsics_shapes: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    pose_overwritten: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compact_summary(self) -> Dict[str, Any]:
        """Lightweight frame snapshot suitable for interactive HTML trajectory playback."""
        return {
            "step": self.step,
            "t": round(self.time_s, 4),
            "arms": {
                k: {
                    "agent": a.agent_id,
                    "pos": [round(float(x), 4) for x in a.ee_pos],
                    "g": round(float(a.gripper_open), 3),
                    "ik": bool(a.ik_valid),
                }
                for k, a in self.arms.items()
            },
            "bodies": {
                k: {
                    "pos": [round(float(x), 4) for x in b.pos],
                    "speed": round(b.linear_speed, 4),
                    "tilt": round(float(b.tilt_deg), 2),
                    "joint": round(float(b.joint_deg), 2),
                    "regions": list(b.in_regions),
                    "fixture": bool(b.is_fixture),
                    "container": b.container,
                }
                for k, b in self.bodies.items()
            },
            "contacts": [
                {
                    "a": c.entity_a,
                    "b": c.entity_b,
                    "grasp": bool(c.whitelisted_grasp),
                    "f": round(float(c.normal_force_n), 2),
                }
                for c in self.contacts
            ],
            "meta": dict(self.metadata),
        }


@dataclass
class ExecutionMode:
    """Execution condition for nominal team rollout vs Appendix A operational probes."""

    mode: str = "team"  # team | single_<aid> | relax_O | relax_W | relax_U | serialize | delay | role_swap
    primary_agent: str = "auto"
    delay_s: float = 0.0
    seed: int = 0

    def label(self) -> str:
        if self.mode == "delay":
            return f"delay:{self.delay_s:g}s"
        if self.mode.startswith("relax_") and self.primary_agent not in ("", "auto"):
            return f"{self.mode}({self.primary_agent})"
        return self.mode


@dataclass
class EpisodeTrace:
    """Full trajectory output of one candidate demonstration or probe run."""

    task: str
    seed: int
    dt_s: float
    frames: List[ControlFrame]
    initial_bodies: Dict[str, BodyState]
    execution_mode: ExecutionMode = field(default_factory=ExecutionMode)
    planner_feasible: bool = True
    planner_failure_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.frames[-1].time_s if self.frames else 0.0

    def sample_keyframes(self, max_frames: int = 90) -> List[Dict[str, Any]]:
        if not self.frames:
            return []
        if len(self.frames) <= max_frames:
            return [fr.compact_summary() for fr in self.frames]
        step = max(1, len(self.frames) // max_frames)
        picked = self.frames[::step]
        if picked[-1].step != self.frames[-1].step:
            picked.append(self.frames[-1])
        return [fr.compact_summary() for fr in picked]


def effective_capabilities(
    spec: TaskSpec, mode: ExecutionMode
) -> Tuple[List[str], Dict[str, EmbodimentSpec], Dict[str, str]]:
    """General N-agent helper computing active agents, relaxed capabilities, and role swaps.

    Used by generated `expert.py` scripts to faithfully execute Appendix A probes:
      - `team`: all agents active with nominal capabilities
      - `single_<aid>`: only `<aid>` active with its own (O_k, W_k, U_k)
      - `relax_O`: primary agent gains union of all team observable zones O
      - `relax_W`: primary agent additionally gains union workspace bounds W
      - `relax_U`: primary agent additionally gains combined arms & payload U
      - `role_swap`: cyclically permutes agent role assignments (for heterogeneous teams)
    """
    aids = spec.agent_ids
    caps: Dict[str, EmbodimentSpec] = {}
    for aid, emb in spec.team.items():
        caps[aid] = EmbodimentSpec(
            agent_id=aid,
            robot=emb.robot,
            arms=emb.arms,
            action_dim=emb.action_dim,
            views=list(emb.views),
            urdf=emb.urdf,
            dofs_per_arm=emb.dofs_per_arm,
            action30_slots=list(emb.action30_slots),
            workspace_bounds=[list(b) for b in emb.workspace_bounds],
            observable_zones=list(emb.observable_zones),
            payload_kg=emb.payload_kg,
        )

    role_map = {aid: aid for aid in aids}
    m = mode.mode
    if m == "team" or m in ("serialize", "delay"):
        return list(aids), caps, role_map
    if m.startswith("single_"):
        target = m[len("single_") :]
        if target not in caps:
            raise ValueError(f"Unknown single-agent mode {m!r}; team agents are {aids}")
        return [target], caps, role_map
    if m in ("relax_O", "relax_W", "relax_U"):
        primary = mode.primary_agent if mode.primary_agent in caps else aids[0]
        eb = caps[primary]
        others = [caps[k] for k in aids if k != primary]
        all_obs = set(eb.observable_zones)
        for o in others:
            all_obs |= set(o.observable_zones)
        eb.observable_zones = sorted(all_obs)
        if m in ("relax_W", "relax_U"):
            lo = [min([eb.workspace_bounds[0][i]] + [o.workspace_bounds[0][i] for o in others]) for i in range(3)]
            hi = [max([eb.workspace_bounds[1][i]] + [o.workspace_bounds[1][i] for o in others]) for i in range(3)]
            eb.workspace_bounds = [lo, hi]
        if m == "relax_U":
            eb.arms = eb.arms + sum(o.arms for o in others)
            eb.payload_kg = eb.payload_kg + sum(o.payload_kg for o in others)
        return [primary], caps, role_map
    if m == "role_swap":
        swapped = {aids[i]: aids[(i + 1) % len(aids)] for i in range(len(aids))}
        return list(aids), caps, swapped
    return list(aids), caps, role_map
