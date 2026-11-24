"""Deterministic evaluator for Task 6: Lift tray, pour food (Sync / Cap / Homo).

Milestones (6 total):
  m_A1 / m_B1: Robot A / B grasps own side of tray with both hands and lifts >= 60 mm.
  m_A2 / m_B2: After lifting, Robot A / B maintains bimanual grasp at tilt >= 5 deg.
  c_1:         Four-gripper airborne carry retains all 3 food items until pouring begins.
  c_2:         All 3 items (apple, carrot, cucumber) settle inside the bag at rest.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ma_bench.evaluators.base import BaseTaskEvaluator, StepEvidence, external_support, is_slow, prefix_contacts_body
from ma_bench.metrics import TrialMetrics, compute_trial_metrics


class LiftTrayPourFoodEvaluator(BaseTaskEvaluator):
    task_slug = "lift_tray_pour_food"

    def __init__(
        self,
        seed: int = 0,
        lift_min_m: float = 0.060,
        min_tilt_deg: float = 5.0,
        food_items: Sequence[str] = ("apple", "carrot", "cucumber"),
    ) -> None:
        super().__init__(seed=seed)
        self.lift_min_m = lift_min_m
        self.min_tilt_deg = min_tilt_deg
        self.food_items = tuple(food_items)

        self._init_tray_z: Optional[float] = None
        self.bits = {"m_A1": False, "m_A2": False, "m_B1": False, "m_B2": False}
        self.shared_carry_seen = False
        self.premature_drop = False
        self.all_in_bag_at_end = False

    def update(self, step: StepEvidence) -> None:
        self.steps += 1
        self.time_s = step.time_s
        tray = step.objects["tray"]
        pos = np.asarray(tray["pos"], dtype=float)
        if self._init_tray_z is None:
            self._init_tray_z = float(pos[2])

        lift = pos[2] - self._init_tray_z
        tilt = abs(float(tray.get("tilt_deg", 0.0)))
        airborne = not external_support(step.contacts, "tray", support_names=("table",))

        both_sides: Dict[str, bool] = {}
        for side, lp, rp in (("A", "robot_a_left", "robot_a_right"), ("B", "robot_b_left", "robot_b_right")):
            alt_l, alt_r = f"{side}.left", f"{side}.right"
            both = (prefix_contacts_body(step.contacts, lp, "tray") or prefix_contacts_body(step.contacts, alt_l, "tray")) and (
                prefix_contacts_body(step.contacts, rp, "tray") or prefix_contacts_body(step.contacts, alt_r, "tray")
            )
            both_sides[side] = both
            if both and lift >= self.lift_min_m - 1e-6:
                self.bits[f"m_{side}1"] = True
            if self.bits[f"m_{side}1"] and both and tilt >= self.min_tilt_deg - 1e-6:
                self.bits[f"m_{side}2"] = True

        if both_sides["A"] and both_sides["B"] and airborne:
            if all(step.objects.get(it, {}).get("container") == "tray" for it in self.food_items):
                self.shared_carry_seen = True
            if tilt < self.min_tilt_deg:
                if any(step.objects.get(it, {}).get("container") not in ("tray", "bag") for it in self.food_items):
                    self.premature_drop = True

        self.all_in_bag_at_end = all(
            step.objects.get(it, {}).get("container") == "bag" and is_slow(step.objects.get(it, {}))
            for it in self.food_items
        )

    def report(self, execution_valid: bool = True) -> TrialMetrics:
        c1 = bool(self.shared_carry_seen and not self.premature_drop)
        c2 = bool(c1 and self.all_in_bag_at_end)
        return compute_trial_metrics(
            task=self.task_slug,
            seed=self.seed,
            role_A_flags={"m_A1": self.bits["m_A1"], "m_A2": self.bits["m_A2"]},
            role_B_flags={"m_B1": self.bits["m_B1"], "m_B2": self.bits["m_B2"]},
            collab_flags={"c_1": c1, "c_2": c2},
            execution_valid=execution_valid,
            details={"premature_drop": self.premature_drop},
        )
