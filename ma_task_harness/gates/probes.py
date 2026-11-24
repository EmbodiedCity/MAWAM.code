"""Task-agnostic Appendix A operational probe arbiter for any N >= 2 TaskSpec.

Certifies:
  - Multi-agent necessity (Eq. 2): team passes AND every single_<aid> fails
  - Why (Eq. 3): minimal relaxation among O < W < U restoring single-agent goal V
  - When: serialize -> asynchronous; all delays pass -> partial; breaks <= tol -> synchronous
  - Who: intrinsic robot descriptor hashes + role_swap infeasibility on heterogeneous teams
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ma_task_harness.spec.schema import TaskSpec


@dataclass
class ProbeTrial:
    mode: str
    success: bool
    planner_feasible: bool
    physical_passed: bool = False
    goal_reached: bool = False
    joint_success: bool = False
    S_role: float = 0.0
    S_collab: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "success": self.success,
            "planner_feasible": self.planner_feasible,
            "physical_passed": self.physical_passed,
            "goal_reached": self.goal_reached,
            "joint_success": self.joint_success,
            "S_role": self.S_role,
            "S_collab": self.S_collab,
            "reason": self.reason,
        }


@dataclass
class ProbeReport:
    necessity_holds: bool = False
    why_certified: Optional[str] = None
    when_certified: Optional[str] = None
    who_certified: Optional[str] = None
    matches_labels: bool = False
    declared_labels: Dict[str, str] = field(default_factory=dict)
    trials: List[ProbeTrial] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "necessity_holds": self.necessity_holds,
            "why_certified": self.why_certified,
            "when_certified": self.when_certified,
            "who_certified": self.who_certified,
            "matches_labels": self.matches_labels,
            "declared_labels": dict(self.declared_labels),
            "trials": [t.to_dict() for t in self.trials],
            "notes": list(self.notes),
        }


def certify_probes(spec: TaskSpec, trials: Sequence[ProbeTrial]) -> ProbeReport:
    """Deterministically derive Why/When/Who labels from executed probe trials."""
    by_mode = {t.mode: t for t in trials}
    report = ProbeReport(declared_labels=spec.labels.to_dict(), trials=list(trials))

    team = by_mode.get("team")
    singles = [by_mode.get(f"single_{aid}") for aid in spec.agent_ids]
    team_ok = bool(team and team.success)
    all_singles_fail = all((s is not None and not s.success) for s in singles)
    report.necessity_holds = team_ok and all_singles_fail
    if report.necessity_holds:
        report.notes.append(f"Eq.(2) necessity holds: team succeeds while all {len(spec.agent_ids)} single agents fail.")
    else:
        report.notes.append(
            f"Eq.(2) necessity failed: team={team_ok}, singles={[s.success if s else None for s in singles]}"
        )

    # Why: first restoring relaxation in O < W < U
    kappa: Optional[str] = None
    for pname, tag in (("relax_O", "information"), ("relax_W", "spatial"), ("relax_U", "capability")):
        tr = by_mode.get(pname)
        if tr and tr.success and kappa is None:
            kappa = tag
    report.why_certified = kappa
    if kappa is None:
        report.notes.append("No relaxation in O < W < U restored single-agent goal feasibility.")
    else:
        report.notes.append(f"Eq.(3) minimal restoring relaxation: kappa = {kappa}.")

    # When: serialize then delay scan
    ser = by_mode.get("serialize")
    if ser and ser.success:
        report.when_certified = "asynchronous"
        report.notes.append("Serialization succeeds => asynchronous.")
    else:
        delays: List[tuple[float, bool]] = []
        for d in spec.probes.delay_seconds:
            lbl = f"delay:{d:g}s"
            tr = by_mode.get(lbl)
            if tr is not None:
                delays.append((float(d), bool(tr.success)))
        if delays and all(ok for _, ok in delays):
            report.when_certified = "partial"
            report.notes.append("Success persists across all waiting delays => partially synchronous.")
        else:
            break_at = next((d for d, ok in delays if not ok), None)
            tol = spec.probes.sync_break_tolerance_s
            if break_at is not None and break_at <= tol + 1e-9:
                report.when_certified = "synchronous"
                report.notes.append(f"Success breaks at delta={break_at}s <= {tol}s => synchronous.")
            elif any(ok for _, ok in delays):
                report.when_certified = "partial"
                report.notes.append("Tolerates moderate waiting delays => partially synchronous.")
            else:
                report.when_certified = "synchronous"
                report.notes.append("All delay probes fail after serialization fails => synchronous.")

    # Who: descriptor hash + role_swap on hetero
    hashes = {ag.descriptor_hash() for ag in spec.team.values()}
    inferred = "homogeneous" if len(hashes) == 1 else "heterogeneous"
    if inferred == "heterogeneous":
        swap = by_mode.get("role_swap")
        if swap is not None and not swap.success:
            report.who_certified = "heterogeneous"
            report.notes.append("Role-swap infeasible => embodiment-imposed heterogeneity certified.")
        else:
            report.who_certified = "heterogeneous_unverified_swap"
            report.notes.append("Heterogeneous descriptors, but role_swap succeeded or was not run.")
    else:
        report.who_certified = "homogeneous"
        report.notes.append("Identical robot descriptors => homogeneous.")

    report.matches_labels = (
        report.necessity_holds
        and report.why_certified == spec.labels.why
        and report.when_certified == spec.labels.when
        and report.who_certified == spec.labels.who
    )
    if not report.matches_labels:
        report.notes.append(
            f"Label mismatch: declared={spec.labels.to_dict()} vs certified="
            f"{{why={report.why_certified}, when={report.when_certified}, who={report.who_certified}, necessity={report.necessity_holds}}}"
        )
    return report
