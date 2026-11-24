"""Task-agnostic Eq. (4)–(5) milestone aggregator and evaluator contract verifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from ma_task_harness.spec.schema import TaskSpec


@dataclass
class MilestoneReport:
    goal_flags: Dict[str, bool] = field(default_factory=dict)
    role_flags: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    collab_flags: Dict[str, bool] = field(default_factory=dict)
    role_scores: Dict[str, float] = field(default_factory=dict)
    S_role: float = 0.0
    S_collab: float = 0.0
    goal_reached: bool = False
    joint_success: bool = False
    contract_errors: List[str] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        flat_flags: Dict[str, bool] = {}
        flat_flags.update(self.goal_flags)
        for d in self.role_flags.values():
            flat_flags.update(d)
        flat_flags.update(self.collab_flags)
        return {
            "flags": flat_flags,
            "goal_flags": dict(self.goal_flags),
            "role_flags": {k: dict(v) for k, v in self.role_flags.items()},
            "collab_flags": dict(self.collab_flags),
            "role_scores": dict(self.role_scores),
            "S_role": self.S_role,
            "S_collab": self.S_collab,
            "goal_reached": self.goal_reached,
            "joint_success": self.joint_success,
            "contract_errors": list(self.contract_errors),
            "notes": dict(self.notes),
        }


def aggregate_evaluator_output(spec: TaskSpec, raw: Mapping[str, Any]) -> MilestoneReport:
    """Validate an evaluator's raw boolean flag dict against `TaskSpec` and compute Eqs. (4)–(5).

    Supports any N >= 2 agents declared in `spec.team`. Rejects missing predicate IDs
    or non-boolean values so an evaluator cannot silently drop milestones.
    """
    rep = MilestoneReport()
    if not isinstance(raw, Mapping):
        rep.contract_errors.append(f"Evaluator must return a dict, got {type(raw).__name__}")
        return rep

    flags_src: Mapping[str, Any] = raw.get("flags", raw)
    if not isinstance(flags_src, Mapping):
        rep.contract_errors.append("Evaluator `flags` must be a dict of predicate_id -> bool")
        return rep

    def _extract(pred_id: str) -> bool:
        if pred_id not in flags_src:
            rep.contract_errors.append(f"Evaluator omitted required predicate {pred_id!r}")
            return False
        val = flags_src[pred_id]
        if not isinstance(val, bool):
            rep.contract_errors.append(f"Predicate {pred_id!r} must be bool, got {type(val).__name__}")
            return False
        return val

    for g in spec.goal:
        rep.goal_flags[g.id] = _extract(g.id)

    for aid in spec.agent_ids:
        rep.role_flags[aid] = {}
        for m in spec.role_milestones[aid]:
            rep.role_flags[aid][m.id] = _extract(m.id)
        vals = list(rep.role_flags[aid].values())
        rep.role_scores[aid] = sum(vals) / max(1, len(vals))

    for c in spec.collab_milestones:
        rep.collab_flags[c.id] = _extract(c.id)

    rep.S_role = sum(rep.role_scores.values()) / max(1, len(rep.role_scores))
    c_vals = list(rep.collab_flags.values())
    rep.S_collab = sum(c_vals) / max(1, len(c_vals))
    rep.goal_reached = bool(rep.goal_flags) and all(rep.goal_flags.values()) and not rep.contract_errors
    all_roles = all(all(d.values()) for d in rep.role_flags.values())
    all_collab = all(rep.collab_flags.values())
    rep.joint_success = bool(rep.goal_reached and all_roles and all_collab and not rep.contract_errors)
    if isinstance(raw.get("notes"), Mapping):
        rep.notes = dict(raw["notes"])
    return rep
