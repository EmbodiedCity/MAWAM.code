"""Milestone metrics (Eqs. 4-5) and 95% Wilson score confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Mapping, Sequence, Tuple


@dataclass
class TrialMetrics:
    task: str
    seed: int
    milestones: Dict[str, bool]
    S_A: float
    S_B: float
    S_role: float
    S_collab: float
    joint_success: bool
    execution_valid: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "seed": self.seed,
            "milestones": dict(self.milestones),
            "S_A": self.S_A,
            "S_B": self.S_B,
            "S_role": self.S_role,
            "S_collab": self.S_collab,
            "joint_success": self.joint_success,
            "execution_valid": self.execution_valid,
            "details": dict(self.details),
        }


def compute_trial_metrics(
    task: str,
    seed: int,
    role_A_flags: Mapping[str, bool],
    role_B_flags: Mapping[str, bool],
    collab_flags: Mapping[str, bool],
    extra_goal_ok: bool = True,
    execution_valid: bool = True,
    details: Mapping[str, Any] | None = None,
) -> TrialMetrics:
    """Compute per-trial Role Score S_k, Collaboration Score S_collab, and joint SR (Eqs. 4-5)."""
    if not role_A_flags or not role_B_flags or not collab_flags:
        raise ValueError("Role A, Role B, and Collaboration milestone dicts must be non-empty")
    s_a = sum(bool(v) for v in role_A_flags.values()) / len(role_A_flags)
    s_b = sum(bool(v) for v in role_B_flags.values()) / len(role_B_flags)
    s_role = 0.5 * (s_a + s_b)
    s_collab = sum(bool(v) for v in collab_flags.values()) / len(collab_flags)
    all_flags = {
        **{k: bool(v) for k, v in role_A_flags.items()},
        **{k: bool(v) for k, v in role_B_flags.items()},
        **{k: bool(v) for k, v in collab_flags.items()},
    }
    joint = bool(execution_valid and extra_goal_ok and all(all_flags.values()))
    return TrialMetrics(
        task=task,
        seed=seed,
        milestones=all_flags,
        S_A=s_a,
        S_B=s_b,
        S_role=s_role,
        S_collab=s_collab,
        joint_success=joint,
        execution_valid=bool(execution_valid),
        details=dict(details or {}),
    )


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """95% Wilson score confidence interval (returned in [0, 100] percent)."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    lo = max(0.0, center - half) * 100.0
    hi = min(1.0, center + half) * 100.0
    return (round(lo, 1), round(hi, 1))


def aggregate_task_metrics(trials: Sequence[TrialMetrics]) -> Dict[str, Any]:
    """Aggregate E independent trajectory trials into paper Table 2 / Table 8 percentages."""
    n = len(trials)
    if n == 0:
        return {"trials": 0, "S_A": 0.0, "S_B": 0.0, "S_role": 0.0, "S_collab": 0.0, "SR": 0.0, "SR_ci95": [0.0, 0.0]}
    s_a = sum(t.S_A for t in trials) / n * 100.0
    s_b = sum(t.S_B for t in trials) / n * 100.0
    s_role = sum(t.S_role for t in trials) / n * 100.0
    s_collab = sum(t.S_collab for t in trials) / n * 100.0
    succ = sum(1 for t in trials if t.joint_success)
    sr = succ / n * 100.0
    lo, hi = wilson_ci(succ, n)
    return {
        "trials": n,
        "successes": succ,
        "S_A": round(s_a, 1),
        "S_B": round(s_b, 1),
        "S_role": round(s_role, 1),
        "S_collab": round(s_collab, 1),
        "SR": round(sr, 1),
        "SR_ci95": [lo, hi],
    }
