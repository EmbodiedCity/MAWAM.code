"""Deterministic evaluator for Task 1: Joint bowl placing (Async / Spatial / Homo).

Milestones (6 total):
  m_A1: Robot A lifts bowl >= 25 mm off start zone without external support.
  m_A2: After m_A1, bowl rests in shared exchange zone (29x25 cm), released and slow.
  m_B1: Robot B lifts bowl >= 25 mm off exchange zone without external support.
  m_B2: After m_B1, bowl rests in goal zone (29x25 cm), released and slow for >= 0.8 s.
  c_1:  Bowl is handed off after A releases with zero simultaneous A/B grasp overlap.
  c_2:  No cross-zone reach into partner's exclusive workspace.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

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


class JointBowlPlacingEvaluator(BaseTaskEvaluator):
    task_slug = "joint_bowl_placing"

    def __init__(
        self,
        seed: int = 0,
        lift_min_m: float = 0.025,
        terminal_rest_s: float = 0.8,
        linear_speed_max: float = 0.02,
        angular_speed_max: float = 0.15,
        exchange_bounds_xy: Tuple[Tuple[float, float], Tuple[float, float]] = ((-0.145, -0.125), (0.145, 0.125)),
        goal_bounds_xy: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.32, -0.125), (0.61, 0.125)),
    ) -> None:
        super().__init__(seed=seed)
        self.lift_min_m = lift_min_m
        self.terminal_rest_s = terminal_rest_s
        self.linear_speed_max = linear_speed_max
        self.angular_speed_max = angular_speed_max
        self.exchange_bounds_xy = exchange_bounds_xy
        self.goal_bounds_xy = goal_bounds_xy
        self._init_z: Optional[float] = None
        self.bits: Dict[str, bool] = {"m_A1": False, "m_A2": False, "m_B1": False, "m_B2": False}
        self.handoff_released_t: Optional[float] = None
        self.handoff_received = False
        self.overlap = False
        self.cross_zone = False
        self.goal_rest_s = 0.0

    @staticmethod
    def _in_rect(xy: np.ndarray, bounds: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
        (x0, y0), (x1, y1) = bounds
        return bool(x0 - 1e-6 <= xy[0] <= x1 + 1e-6 and y0 - 1e-6 <= xy[1] <= y1 + 1e-6)

    def update(self, step: StepEvidence) -> None:
        self.steps += 1
        self.time_s = step.time_s
        bowl = step.objects["bowl"]
        pos = np.asarray(bowl["pos"], dtype=float)
        if self._init_z is None:
            self._init_z = float(pos[2])

        a_hold = prefix_contacts_body(step.contacts, "robot_a", "bowl") or prefix_contacts_body(step.contacts, "A.", "bowl")
        b_hold = prefix_contacts_body(step.contacts, "robot_b", "bowl") or prefix_contacts_body(step.contacts, "B.", "bowl")
        if a_hold and b_hold:
            self.overlap = True

        # Cross-zone boundary check against exchange zone x limits
        a_max_x = self.exchange_bounds_xy[1][0]
        b_min_x = self.exchange_bounds_xy[0][0]
        for hand_key, hp in step.hand_positions.items():
            if (hand_key.startswith("robot_a") or hand_key.startswith("A.")) and hp[0] > a_max_x:
                self.cross_zone = True
            if (hand_key.startswith("robot_b") or hand_key.startswith("B.")) and hp[0] < b_min_x:
                self.cross_zone = True

        supported = external_support(step.contacts, "bowl")
        released = not any_robot_contact(step.contacts, "bowl")
        slow = is_slow(bowl, self.linear_speed_max, self.angular_speed_max)
        lifted = (not supported) and (pos[2] - self._init_z >= self.lift_min_m - 1e-6)

        if a_hold and lifted and not self.bits["m_A1"]:
            self.bits["m_A1"] = True

        in_exchange = self._in_rect(pos[:2], self.exchange_bounds_xy) or "exchange_zone" in bowl.get("in_regions", ())
        if self.bits["m_A1"] and in_exchange and supported and released and slow:
            self.bits["m_A2"] = True
            if self.handoff_released_t is None:
                self.handoff_released_t = step.time_s

        if b_hold and lifted and self.handoff_released_t is not None and step.time_s > self.handoff_released_t:
            self.bits["m_B1"] = True
            self.handoff_received = True

        in_goal = self._in_rect(pos[:2], self.goal_bounds_xy) or "goal_zone" in bowl.get("in_regions", ())
        if self.bits["m_B1"] and in_goal and supported and released and slow:
            self.goal_rest_s += step.dt_s
            if self.goal_rest_s + 1e-9 >= self.terminal_rest_s:
                self.bits["m_B2"] = True
        else:
            self.goal_rest_s = 0.0

    def report(self, execution_valid: bool = True) -> TrialMetrics:
        c1 = bool(self.handoff_received and not self.overlap)
        c2 = bool(self.bits["m_A2"] and self.bits["m_B2"] and not self.cross_zone)
        return compute_trial_metrics(
            task=self.task_slug,
            seed=self.seed,
            role_A_flags={"m_A1": self.bits["m_A1"], "m_A2": self.bits["m_A2"]},
            role_B_flags={"m_B1": self.bits["m_B1"], "m_B2": self.bits["m_B2"]},
            collab_flags={"c_1": c1, "c_2": c2},
            execution_valid=execution_valid,
            details={"overlap": self.overlap, "cross_zone": self.cross_zone},
        )
