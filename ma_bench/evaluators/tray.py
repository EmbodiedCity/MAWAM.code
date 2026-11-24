"""Deterministic evaluator for Task 5: Lift tray, keep blocks (Sync / Cap / Homo).

Milestones (7 total):
  m_A1 / m_B1: Robot A / B grasps tray with both hands while airborne (> 85 mm).
  m_A2 / m_B2: After lifting, Robot A / B sets down and releases tray cleanly for >= 0.8 s.
  c_1:         Four-gripper airborne shared carry lasts >= 0.4 s with >= 20 mm horizontal transport.
  c_2:         Maximum tray tilt stays <= 8 deg across the entire trajectory.
  c_3:         All four stacked blocks remain intact on the tray throughout transport and rest.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ma_bench.evaluators.base import (
    BaseTaskEvaluator,
    StepEvidence,
    any_robot_contact,
    external_support,
    is_slow,
    prefix_contacts_body,
)
from ma_bench.metrics import TrialMetrics, compute_trial_metrics


class LiftTrayKeepBlocksEvaluator(BaseTaskEvaluator):
    task_slug = "lift_tray_keep_blocks"

    def __init__(
        self,
        seed: int = 0,
        lift_min_m: float = 0.085,
        shared_window_min_s: float = 0.40,
        transport_xy_min_m: float = 0.020,
        max_tilt_deg: float = 8.0,
        terminal_tilt_deg: float = 3.0,
        terminal_rest_s: float = 0.80,
        block_names: Sequence[str] = ("block_1", "block_2", "block_3", "block_4"),
    ) -> None:
        super().__init__(seed=seed)
        self.lift_min_m = lift_min_m
        self.shared_window_min_s = shared_window_min_s
        self.transport_xy_min_m = transport_xy_min_m
        self.max_tilt_deg = max_tilt_deg
        self.terminal_tilt_deg = terminal_tilt_deg
        self.terminal_rest_s = terminal_rest_s
        self.block_names = tuple(block_names)

        self._init_tray_z: Optional[float] = None
        self.bits = {"m_A1": False, "m_A2": False, "m_B1": False, "m_B2": False}
        self.rest_s = {"A": 0.0, "B": 0.0}
        self.shared_streak_s = 0.0
        self.shared_xy0: Optional[np.ndarray] = None
        self.valid_shared_carry = False
        self.max_tilt_seen = 0.0
        self.cargo_retained = True

    def update(self, step: StepEvidence) -> None:
        self.steps += 1
        self.time_s = step.time_s
        tray = step.objects["tray"]
        pos = np.asarray(tray["pos"], dtype=float)
        if self._init_tray_z is None:
            self._init_tray_z = float(pos[2])

        tilt = abs(float(tray.get("tilt_deg", 0.0)))
        self.max_tilt_seen = max(self.max_tilt_seen, tilt)

        for bn in self.block_names:
            b = step.objects.get(bn)
            if b is None or b.get("container") != "tray":
                self.cargo_retained = False

        supported = external_support(step.contacts, "tray", support_names=("table",))
        airborne = (not supported) and (pos[2] - self._init_tray_z > self.lift_min_m)
        slow = is_slow(tray)

        grips: Dict[str, bool] = {}
        for side, lp, rp in (("A", "robot_a_left", "robot_a_right"), ("B", "robot_b_left", "robot_b_right")):
            alt_l, alt_r = f"{side}.left", f"{side}.right"
            both = (prefix_contacts_body(step.contacts, lp, "tray") or prefix_contacts_body(step.contacts, alt_l, "tray")) and (
                prefix_contacts_body(step.contacts, rp, "tray") or prefix_contacts_body(step.contacts, alt_r, "tray")
            )
            grips[side] = both
            if both and airborne:
                self.bits[f"m_{side}1"] = True

            released = (not prefix_contacts_body(step.contacts, f"robot_{side.lower()}", "tray")) and (
                not prefix_contacts_body(step.contacts, f"{side}.", "tray")
            )
            target_ok = ("tray_goal_zone" in tray.get("in_regions", ())) and (tilt < self.terminal_tilt_deg)
            good = self.bits[f"m_{side}1"] and target_ok and supported and slow and released
            self.rest_s[side] = (self.rest_s[side] + step.dt_s) if good else 0.0
            if self.rest_s[side] + 1e-9 >= self.terminal_rest_s:
                self.bits[f"m_{side}2"] = True

        if grips["A"] and grips["B"] and airborne:
            self.shared_streak_s += step.dt_s
            if self.shared_xy0 is None:
                self.shared_xy0 = pos[:2].copy()
            dist_xy = float(np.linalg.norm(pos[:2] - self.shared_xy0))
            if self.shared_streak_s + 1e-9 >= self.shared_window_min_s and dist_xy >= self.transport_xy_min_m - 1e-6:
                self.valid_shared_carry = True
        else:
            self.shared_streak_s = 0.0
            self.shared_xy0 = None

    def report(self, execution_valid: bool = True) -> TrialMetrics:
        c1 = bool(self.valid_shared_carry)
        c2 = bool(self.max_tilt_seen <= self.max_tilt_deg + 1e-6)
        c3 = bool(self.cargo_retained)
        return compute_trial_metrics(
            task=self.task_slug,
            seed=self.seed,
            role_A_flags={"m_A1": self.bits["m_A1"], "m_A2": self.bits["m_A2"]},
            role_B_flags={"m_B1": self.bits["m_B1"], "m_B2": self.bits["m_B2"]},
            collab_flags={"c_1": c1, "c_2": c2, "c_3": c3},
            execution_valid=execution_valid,
            details={"max_tilt_deg": self.max_tilt_seen},
        )
