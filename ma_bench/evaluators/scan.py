"""Deterministic evaluator for Task 3: Hold and scan (Partial / Spatial / Homo).

Milestones (6 total):
  m_A1: Robot A lifts product box >= 25 mm off tabletop.
  m_A2: After cooperative scan, Robot A returns box within 65 mm of start, released, slow >= 0.8 s.
  m_B1: Robot B holds scanner aligned with airborne box within distance & angle tolerance.
  m_B2: After scan and ordered withdrawal, Robot B returns scanner within 65 mm, released, slow >= 0.8 s.
  c_1:  Airborne scan geometry maintained continuously for >= 0.6 s.
  c_2:  Scanner operator B withdraws (>= 30 mm) strictly before holder A lowers box.
"""

from __future__ import annotations

from typing import Dict, Optional

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


class HoldAndScanEvaluator(BaseTaskEvaluator):
    task_slug = "hold_and_scan"

    def __init__(
        self,
        seed: int = 0,
        lift_min_m: float = 0.025,
        scan_dist_max_m: float = 0.14,
        scan_angle_max_deg: float = 20.0,
        scan_hold_s: float = 0.60,
        withdraw_min_m: float = 0.030,
        return_tol_m: float = 0.065,
        terminal_rest_s: float = 0.80,
    ) -> None:
        super().__init__(seed=seed)
        self.lift_min_m = lift_min_m
        self.scan_dist_max_m = scan_dist_max_m
        self.scan_angle_max_deg = scan_angle_max_deg
        self.scan_hold_s = scan_hold_s
        self.withdraw_min_m = withdraw_min_m
        self.return_tol_m = return_tol_m
        self.terminal_rest_s = terminal_rest_s

        self._init: Dict[str, np.ndarray] = {}
        self.bits = {"m_A1": False, "m_A2": False, "m_B1": False, "m_B2": False}
        self.scan_streak_s = 0.0
        self.scan_complete = False
        self.scan_pos: Optional[Dict[str, np.ndarray]] = None
        self.b_withdraw_t: Optional[float] = None
        self.a_withdraw_t: Optional[float] = None
        self.withdrawal_violation = False
        self.return_streak_s = {"tea_box": 0.0, "scanner": 0.0}

    def update(self, step: StepEvidence) -> None:
        self.steps += 1
        self.time_s = step.time_s
        box = step.objects["tea_box"]
        scn = step.objects["scanner"]
        p_box = np.asarray(box["pos"], dtype=float)
        p_scn = np.asarray(scn["pos"], dtype=float)
        if not self._init:
            self._init = {"tea_box": p_box.copy(), "scanner": p_scn.copy()}

        hold_a = prefix_contacts_body(step.contacts, "robot_a", "tea_box") or prefix_contacts_body(step.contacts, "A.", "tea_box")
        hold_b = prefix_contacts_body(step.contacts, "robot_b", "scanner") or prefix_contacts_body(step.contacts, "B.", "scanner")
        box_airborne = not external_support(step.contacts, "tea_box")
        if hold_a and box_airborne and (p_box[2] - self._init["tea_box"][2] >= self.lift_min_m - 1e-6):
            self.bits["m_A1"] = True

        dist = float(np.linalg.norm(p_scn - p_box))
        angle_deg = float(scn.get("angle_to_target_deg", 0.0))
        geom_ok = (dist <= self.scan_dist_max_m + 1e-6) and (angle_deg <= self.scan_angle_max_deg + 1e-6)
        if hold_a and hold_b and box_airborne and geom_ok:
            self.bits["m_B1"] = True
            self.scan_streak_s += step.dt_s
            if self.scan_streak_s + 1e-9 >= self.scan_hold_s and not self.scan_complete:
                self.scan_complete = True
                self.scan_pos = {"tea_box": p_box.copy(), "scanner": p_scn.copy()}
        else:
            self.scan_streak_s = 0.0

        if self.scan_pos is not None:
            d_box = float(np.linalg.norm(p_box - self.scan_pos["tea_box"]))
            d_scn = float(np.linalg.norm(p_scn - self.scan_pos["scanner"]))
            # Evaluate A first so simultaneous movement cannot falsely claim B-first.
            if self.a_withdraw_t is None and d_box >= self.withdraw_min_m:
                self.a_withdraw_t = step.time_s
                if self.b_withdraw_t is None:
                    self.withdrawal_violation = True
            if self.b_withdraw_t is None and d_scn >= self.withdraw_min_m and not geom_ok:
                self.b_withdraw_t = step.time_s

        for name, m_key, req_key in (("tea_box", "m_A2", "m_A1"), ("scanner", "m_B2", "m_B1")):
            obj = step.objects[name]
            p = np.asarray(obj["pos"], dtype=float)
            returned = (
                float(np.linalg.norm(p - self._init[name])) <= self.return_tol_m + 1e-6
                and external_support(step.contacts, name)
                and (not any_robot_contact(step.contacts, name))
                and is_slow(obj)
            )
            eligible = self.scan_complete and self.bits[req_key]
            if name == "scanner":
                eligible = eligible and (self.b_withdraw_t is not None) and (not self.withdrawal_violation)
            self.return_streak_s[name] = (self.return_streak_s[name] + step.dt_s) if (returned and eligible) else 0.0
            if self.return_streak_s[name] + 1e-9 >= self.terminal_rest_s:
                self.bits[m_key] = True

    def report(self, execution_valid: bool = True) -> TrialMetrics:
        c1 = bool(self.scan_complete)
        c2 = bool(
            self.b_withdraw_t is not None
            and self.a_withdraw_t is not None
            and self.b_withdraw_t < self.a_withdraw_t
            and not self.withdrawal_violation
            and execution_valid
        )
        return compute_trial_metrics(
            task=self.task_slug,
            seed=self.seed,
            role_A_flags={"m_A1": self.bits["m_A1"], "m_A2": self.bits["m_A2"]},
            role_B_flags={"m_B1": self.bits["m_B1"], "m_B2": self.bits["m_B2"]},
            collab_flags={"c_1": c1, "c_2": c2},
            execution_valid=execution_valid,
            details={"b_withdraw_t": self.b_withdraw_t, "a_withdraw_t": self.a_withdraw_t},
        )
