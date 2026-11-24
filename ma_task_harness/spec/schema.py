"""Task-agnostic TaskSpec schema (Paper Section 3.1–3.2 & Appendix A–B).

A `TaskSpec` is the single source of truth shared by the generation sub-agent
`G^(n)` and evaluation sub-agent `E^(n)` for *any* multi-robot task:
  - Supports arbitrary team sizes N >= 2 with arbitrary agent IDs (e.g. A, B, C, ...)
  - Supports built-in robot presets (ALOHA, Panda, UR5, Franka FR3, Cobot Magic)
    as well as arbitrary user-defined robot embodiments & URDFs
  - Supports open-ended declarative goal, role, and collaboration predicates
    with named `<threshold>` references refined during co-evolution
  - Enforces Why (O < W < U), When (asynchronous | partial | synchronous), and
    Who (homogeneous | heterogeneous) structural & probe invariants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ma_task_harness.spec.yaml_lite import dumps_yaml, load_yaml_file, loads_yaml, save_yaml_file

COMPLEMENTARITY_ORDER: Tuple[str, str, str] = ("information", "spatial", "capability")
VALID_WHY = set(COMPLEMENTARITY_ORDER)
VALID_WHEN = {"asynchronous", "partial", "synchronous"}
VALID_WHO = {"homogeneous", "heterogeneous"}

ROBOT_PRESETS: Dict[str, Dict[str, Any]] = {
    "aloha": {
        "urdf": "aloha_agilex_dual_6dof",
        "arms": 2,
        "dofs_per_arm": 6,
        "action_dim": 16,
        "action30_slots": list(range(0, 14)) + [28, 29],
        "views": ["head", "left_wrist", "right_wrist"],
        "payload_kg": 3.0,
    },
    "panda": {
        "urdf": "franka_panda_7dof",
        "arms": 1,
        "dofs_per_arm": 7,
        "action_dim": 8,
        "action30_slots": list(range(0, 7)) + [28],
        "views": ["head", "wrist"],
        "payload_kg": 3.0,
    },
    "ur5": {
        "urdf": "universal_robots_ur5_6dof",
        "arms": 1,
        "dofs_per_arm": 6,
        "action_dim": 8,
        "action30_slots": list(range(0, 7)) + [28],
        "views": ["head", "wrist"],
        "payload_kg": 5.0,
    },
    "franka_fr3": {
        "urdf": "franka_fr3_7dof",
        "arms": 1,
        "dofs_per_arm": 7,
        "action_dim": 8,
        "action30_slots": list(range(0, 7)) + [28],
        "views": ["external_left", "external_right", "wrist"],
        "payload_kg": 3.0,
    },
    "cobot_magic": {
        "urdf": "agilex_cobot_magic_dual_6dof",
        "arms": 2,
        "dofs_per_arm": 6,
        "action_dim": 14,
        "action30_slots": list(range(0, 6)) + list(range(7, 13)) + [28, 29],
        "views": ["head", "left_wrist", "right_wrist"],
        "payload_kg": 3.0,
    },
}

_THRESHOLD_REF_RE = re.compile(r"^<([A-Za-z_][A-Za-z0-9_]*)>$")
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class TaskSpecValidationError(ValueError):
    """Raised when a TaskSpec violates structural or taxonomy invariants."""


@dataclass
class EmbodimentSpec:
    """Capability tuple E_k = (O_k, W_k, U_k) for one complete robot agent k."""

    agent_id: str
    robot: str
    arms: int
    action_dim: int
    views: List[str]
    urdf: str = ""
    dofs_per_arm: int = 6
    action30_slots: List[int] = field(default_factory=list)
    workspace_bounds: List[List[float]] = field(
        default_factory=lambda: [[-0.6, -0.6, 0.70], [0.6, 0.6, 1.35]]
    )
    observable_zones: List[str] = field(default_factory=lambda: ["shared", "own"])
    payload_kg: float = 3.0

    def __post_init__(self) -> None:
        preset = ROBOT_PRESETS.get(self.robot.lower(), {})
        if not self.urdf:
            self.urdf = str(preset.get("urdf", f"{self.robot.lower()}_urdf"))
        if not self.action30_slots:
            if preset:
                self.action30_slots = list(preset["action30_slots"])
            elif self.arms == 1:
                self.action30_slots = list(range(0, 7)) + [28]
            else:
                self.action30_slots = list(range(0, 14)) + [28, 29]

    def descriptor_dict(self) -> Dict[str, Any]:
        """Intrinsic body descriptor (excluding base placement) for Who homogeneity check."""
        return {
            "robot": self.robot.lower(),
            "urdf": self.urdf,
            "arms": self.arms,
            "dofs_per_arm": self.dofs_per_arm,
            "action_dim": self.action_dim,
            "action30_slots": list(self.action30_slots),
            "views": list(self.views),
            "payload_kg": self.payload_kg,
        }

    def descriptor_hash(self) -> str:
        blob = json.dumps(self.descriptor_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot": self.robot,
            "urdf": self.urdf,
            "arms": self.arms,
            "dofs_per_arm": self.dofs_per_arm,
            "action_dim": self.action_dim,
            "action30_slots": list(self.action30_slots),
            "views": list(self.views),
            "workspace_bounds": [list(b) for b in self.workspace_bounds],
            "observable_zones": list(self.observable_zones),
            "payload_kg": self.payload_kg,
        }

    @classmethod
    def from_dict(cls, agent_id: str, data: Mapping[str, Any]) -> "EmbodimentSpec":
        robot = str(data["robot"]).lower()
        preset = ROBOT_PRESETS.get(robot, {})
        arms = int(data.get("arms", preset.get("arms", 1)))
        action_dim = int(data.get("action_dim", preset.get("action_dim", arms * 8)))
        views = list(data.get("views", preset.get("views", ["head", "wrist"] if arms == 1 else ["head", "left_wrist", "right_wrist"])))
        return cls(
            agent_id=agent_id,
            robot=robot,
            arms=arms,
            action_dim=action_dim,
            views=views,
            urdf=str(data.get("urdf", preset.get("urdf", f"{robot}_urdf"))),
            dofs_per_arm=int(data.get("dofs_per_arm", preset.get("dofs_per_arm", 6))),
            action30_slots=list(data.get("action30_slots", preset.get("action30_slots", []))),
            workspace_bounds=[
                [float(x) for x in row]
                for row in data.get("workspace_bounds", [[-0.6, -0.6, 0.70], [0.6, 0.6, 1.35]])
            ],
            observable_zones=[str(z) for z in data.get("observable_zones", ["shared", "own"])],
            payload_kg=float(data.get("payload_kg", preset.get("payload_kg", 3.0))),
        )


@dataclass
class TaskLabels:
    why: str   # information | spatial | capability
    when: str  # asynchronous | partial | synchronous
    who: str   # homogeneous | heterogeneous

    def to_dict(self) -> Dict[str, str]:
        return {"why": self.why, "when": self.when, "who": self.who}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskLabels":
        when_raw = str(data["when"]).strip().lower()
        if when_raw in ("partially synchronous", "partially_synchronous", "partial_sync"):
            when_raw = "partial"
        return cls(
            why=str(data["why"]).strip().lower(),
            when=when_raw,
            who=str(data["who"]).strip().lower(),
        )


@dataclass
class PromptsSpec:
    global_prompt: str
    roles: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"global": self.global_prompt}
        for k, v in self.roles.items():
            out[k] = v
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], agent_ids: Sequence[str]) -> "PromptsSpec":
        global_prompt = str(data.get("global", "")).strip()
        roles: Dict[str, str] = {}
        if "roles" in data and isinstance(data["roles"], Mapping):
            for k in agent_ids:
                if k in data["roles"]:
                    roles[k] = str(data["roles"][k]).strip()
        else:
            for k in agent_ids:
                if k in data:
                    roles[k] = str(data[k]).strip()
        return cls(global_prompt=global_prompt, roles=roles)


@dataclass
class PredicateSpec:
    """Open-ended declarative milestone or goal predicate compiled by E^(n)."""

    id: str
    kind: str
    description: str = ""
    agent: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def resolve_params(self, thresholds: Mapping[str, Any]) -> Dict[str, Any]:
        def _resolve(val: Any) -> Any:
            if isinstance(val, str):
                m = _THRESHOLD_REF_RE.match(val.strip())
                if m:
                    tname = m.group(1)
                    if tname not in thresholds:
                        raise TaskSpecValidationError(
                            f"Predicate {self.id!r} references undefined threshold <{tname}>"
                        )
                    return thresholds[tname]
                return val
            if isinstance(val, list):
                return [_resolve(x) for x in val]
            if isinstance(val, dict):
                return {k: _resolve(v) for k, v in val.items()}
            return val

        return {k: _resolve(v) for k, v in self.params.items()}

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"id": self.id, "kind": self.kind}
        if self.description:
            out["description"] = self.description
        if self.agent is not None:
            out["agent"] = self.agent
        if self.params:
            out["params"] = dict(self.params)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], default_agent: Optional[str] = None) -> "PredicateSpec":
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            description=str(data.get("description", "")),
            agent=str(data["agent"]) if data.get("agent") is not None else default_agent,
            params=dict(data.get("params", {})),
        )


@dataclass
class ProbeConfig:
    necessity: List[str] = field(default_factory=list)
    serialize: bool = True
    delay_seconds: List[float] = field(default_factory=lambda: [0.08, 0.8, 2.5])
    sync_break_tolerance_s: float = 0.20
    role_swap: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "necessity": list(self.necessity),
            "temporal": {
                "serialize": self.serialize,
                "delay_seconds": list(self.delay_seconds),
                "sync_break_tolerance_s": self.sync_break_tolerance_s,
            },
            "embodiment": ["role_swap"] if self.role_swap else [],
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        agent_ids: Sequence[str],
        is_hetero: bool = False,
    ) -> "ProbeConfig":
        default_nec = [f"single_{aid}" for aid in agent_ids] + ["relax_O", "relax_W", "relax_U"]
        nec = [str(x) for x in data.get("necessity", default_nec)]
        temporal = data.get("temporal", {})
        if isinstance(temporal, Mapping):
            serialize = bool(temporal.get("serialize", True))
            delays = [float(x) for x in temporal.get("delay_seconds", [0.08, 0.8, 2.5])]
            sync_tol = float(temporal.get("sync_break_tolerance_s", 0.20))
        else:
            serialize = True
            delays = [0.08, 0.8, 2.5]
            sync_tol = 0.20
        emb = data.get("embodiment", ["role_swap"] if is_hetero else [])
        role_swap = ("role_swap" in emb) if isinstance(emb, list) else bool(emb)
        return cls(
            necessity=nec,
            serialize=serialize,
            delay_seconds=delays,
            sync_break_tolerance_s=sync_tol,
            role_swap=role_swap,
        )


@dataclass
class TaskSpec:
    """Task-agnostic machine-readable contract for any N-robot collaborative task."""

    task: str
    title: str
    version: str
    team: Dict[str, EmbodimentSpec]
    labels: TaskLabels
    prompts: PromptsSpec
    thresholds: Dict[str, float]
    goal: List[PredicateSpec]
    role_milestones: Dict[str, List[PredicateSpec]]
    collab_milestones: List[PredicateSpec]
    probes: ProbeConfig
    description: str = ""
    nominal_duration_s: float = 40.0
    control_hz: float = 25.0
    physics_hz: float = 250.0
    scene_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def agent_ids(self) -> List[str]:
        return list(self.team.keys())

    def validate(self) -> None:
        """Enforce task-agnostic formal invariants from Section 3.1–3.2 and Appendix A–B."""
        if not self.task or not _SLUG_RE.match(self.task):
            raise TaskSpecValidationError(f"Invalid task slug {self.task!r}; expected snake_case")
        if len(self.team) < 2:
            raise TaskSpecValidationError("Multi-agent TaskSpec requires at least 2 agents in `team`.")

        for aid in self.team:
            if not _ID_RE.match(aid):
                raise TaskSpecValidationError(f"Invalid agent_id {aid!r}")

        if self.labels.why not in VALID_WHY:
            raise TaskSpecValidationError(f"Invalid `why` label {self.labels.why!r}; expected {sorted(VALID_WHY)}")
        if self.labels.when not in VALID_WHEN:
            raise TaskSpecValidationError(f"Invalid `when` label {self.labels.when!r}; expected {sorted(VALID_WHEN)}")
        if self.labels.who not in VALID_WHO:
            raise TaskSpecValidationError(f"Invalid `who` label {self.labels.who!r}; expected {sorted(VALID_WHO)}")

        hashes = {ag.descriptor_hash() for ag in self.team.values()}
        inferred_who = "homogeneous" if len(hashes) == 1 else "heterogeneous"
        if self.labels.who != inferred_who:
            raise TaskSpecValidationError(
                f"Declared labels.who={self.labels.who!r} contradicts robot descriptors ({inferred_who!r})."
            )

        for aid, emb in self.team.items():
            if emb.arms < 1:
                raise TaskSpecValidationError(f"Agent {aid}: arms must be >= 1, got {emb.arms}")
            if emb.action_dim < 1:
                raise TaskSpecValidationError(f"Agent {aid}: action_dim must be >= 1, got {emb.action_dim}")
            if not emb.views:
                raise TaskSpecValidationError(f"Agent {aid}: at least 1 camera view is required")

        if not self.prompts.global_prompt:
            raise TaskSpecValidationError("Prompts must include a non-empty `global` instruction.")
        for aid in self.team:
            if not self.prompts.roles.get(aid):
                raise TaskSpecValidationError(f"Prompts missing local role instruction for agent {aid!r}.")

        if not self.goal:
            raise TaskSpecValidationError("TaskSpec requires at least one terminal `goal` predicate.")
        for aid in self.team:
            if not self.role_milestones.get(aid):
                raise TaskSpecValidationError(f"Agent {aid!r} must have at least one role milestone.")
        if not self.collab_milestones:
            raise TaskSpecValidationError("TaskSpec must declare at least one `collab_milestone`.")

        seen_ids: set[str] = set()
        all_preds: List[PredicateSpec] = list(self.goal) + list(self.collab_milestones)
        for mlist in self.role_milestones.values():
            all_preds.extend(mlist)

        for pred in all_preds:
            if not _ID_RE.match(pred.id):
                raise TaskSpecValidationError(f"Invalid predicate id {pred.id!r}")
            if not _SLUG_RE.match(pred.kind):
                raise TaskSpecValidationError(f"Predicate kind {pred.kind!r} must be snake_case")
            if pred.id in seen_ids:
                raise TaskSpecValidationError(f"Duplicate predicate id {pred.id!r} in TaskSpec {self.task!r}.")
            seen_ids.add(pred.id)
            pred.resolve_params(self.thresholds)

        expected_singles = [f"single_{aid}" for aid in self.agent_ids]
        for s_name in expected_singles:
            if s_name not in self.probes.necessity:
                raise TaskSpecValidationError(
                    f"Necessity probes must include {s_name!r} for every agent in team {self.agent_ids}"
                )
        relax_seq = [x for x in self.probes.necessity if x.startswith("relax_")]
        if relax_seq != ["relax_O", "relax_W", "relax_U"]:
            raise TaskSpecValidationError(
                f"Relaxation probes must appear in strict O < W < U order ['relax_O', 'relax_W', 'relax_U'], got {relax_seq}"
            )
        if self.labels.who == "heterogeneous" and not self.probes.role_swap:
            raise TaskSpecValidationError(
                "Heterogeneous tasks must enable the `role_swap` embodiment probe."
            )

    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "task": self.task,
            "title": self.title,
            "version": self.version,
        }
        if self.description:
            out["description"] = self.description
        out.update(
            {
                "nominal_duration_s": self.nominal_duration_s,
                "control_hz": self.control_hz,
                "physics_hz": self.physics_hz,
                "team": {aid: emb.to_dict() for aid, emb in self.team.items()},
                "labels": self.labels.to_dict(),
                "prompts": self.prompts.to_dict(),
                "thresholds": dict(self.thresholds),
                "scene_params": dict(self.scene_params),
                "goal": [p.to_dict() for p in self.goal],
                "role_milestones": {
                    aid: [p.to_dict() for p in preds] for aid, preds in self.role_milestones.items()
                },
                "collab_milestones": [p.to_dict() for p in self.collab_milestones],
                "probes": self.probes.to_dict(),
            }
        )
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskSpec":
        team_raw = data.get("team", {})
        if not isinstance(team_raw, Mapping) or len(team_raw) < 2:
            raise TaskSpecValidationError("`team` must be a mapping with at least 2 agents.")
        team = {str(aid): EmbodimentSpec.from_dict(str(aid), spec) for aid, spec in team_raw.items()}
        agent_ids = list(team.keys())
        labels = TaskLabels.from_dict(data["labels"])
        prompts = PromptsSpec.from_dict(data["prompts"], agent_ids)
        thresholds = {str(k): float(v) for k, v in data.get("thresholds", {}).items()}
        goal = [PredicateSpec.from_dict(p) for p in data.get("goal", [])]
        role_milestones = {
            str(aid): [PredicateSpec.from_dict(p, default_agent=str(aid)) for p in preds]
            for aid, preds in data.get("role_milestones", {}).items()
        }
        collab_milestones = [PredicateSpec.from_dict(p) for p in data.get("collab_milestones", [])]
        probes = ProbeConfig.from_dict(
            data.get("probes", {}), agent_ids=agent_ids, is_hetero=(labels.who == "heterogeneous")
        )
        spec = cls(
            task=str(data["task"]),
            title=str(data.get("title", data["task"])),
            version=str(data.get("version", "v1")),
            description=str(data.get("description", "")),
            team=team,
            labels=labels,
            prompts=prompts,
            thresholds=thresholds,
            goal=goal,
            role_milestones=role_milestones,
            collab_milestones=collab_milestones,
            probes=probes,
            nominal_duration_s=float(data.get("nominal_duration_s", 40.0)),
            control_hz=float(data.get("control_hz", 25.0)),
            physics_hz=float(data.get("physics_hz", 250.0)),
            scene_params=dict(data.get("scene_params", {})),
        )
        spec.validate()
        return spec

    def to_yaml(self) -> str:
        return dumps_yaml(self.to_dict())


def load_task_spec(path: str | Path) -> TaskSpec:
    raw = load_yaml_file(path)
    if not isinstance(raw, Mapping):
        raise TaskSpecValidationError(f"Expected mapping at root of {path}")
    return TaskSpec.from_dict(raw)


def load_task_spec_from_text(text: str) -> TaskSpec:
    raw = loads_yaml(text)
    if not isinstance(raw, Mapping):
        raise TaskSpecValidationError("Expected mapping at root of TaskSpec YAML")
    return TaskSpec.from_dict(raw)


def save_task_spec(spec: TaskSpec, path: str | Path) -> None:
    spec.validate()
    save_yaml_file(spec.to_dict(), path)


def examples_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "examples" / "mabench"


def load_example_specs() -> Dict[str, TaskSpec]:
    specs: Dict[str, TaskSpec] = {}
    if examples_dir().is_dir():
        for p in sorted(examples_dir().glob("*.yaml")):
            if p.name.startswith("."):
                continue
            spec = load_task_spec(p)
            specs[spec.task] = spec
    return specs
