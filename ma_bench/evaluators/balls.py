"""Deterministic evaluator for Task 7: Pour and catch (Sync / Cap / Hetero).

Milestones (6 total):
  m_A1: Panda contacts tipping hopper during pouring (> 10 deg).
  m_A2: Terminal hopper returns upright (< 5 deg) and Panda gripper opens (> 0.8).
  m_B1: UR5 contacts receiving cup airborne (no external support) during pouring.
  m_B2: Terminal cup rests at target (< 4 cm, < 2 cm/s, < 10 deg) and UR5 releases (> 0.8).
  c_1:  >= 25 cumulative bilateral pouring frames with airborne cup tracking and zero collision.
  c_2:  All 3 balls caught and stable inside the receiving cup across the final 25 frames (1.0 s).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ma_bench.evaluators.base import BaseTaskEvaluator, StepEvidence, external_support, is_slow, prefix_contacts_body
from ma_bench.metrics import TrialMetrics, compute_trial_metrics


class PourAndCatchEvaluator(BaseTaskEvaluator):
    task_slug = "pour_and_catch"

    def __init__(
        self,
        seed: int = 0,
        pour_deg: float = 10.0,
        return_deg: float = 5.0,
        min_bilateral_frames: int = 25,
        terminal_frames: int = 25,
        ball_names: Sequence[str] = ("ball_1", "ball_2", "ball_3"),
    ) -> None:
        super().__init__(seed=seed)
        self.pour_deg = pour_deg
        self.return_deg = return_deg
        self.min_bilateral_frames = min_bilateral_frames
        self.terminal_frames = terminal_frames
        self.ball_names = tuple(ball_names)

        self.m_A1 = False
        self.ur5_tip_contact = False
        self.cup_supported_during_pour = False
        self.bilateral_tip_frames = 0
        self.robot_collision = False
        self.all_balls_in_history: List[bool] = []
        self._last_step: StepEvidence | None = None

    def update(self, step: StepEvidence) -> None:
        self.steps += 1
        self.time_s = step.time_s
        self._last_step = step
        if step.robot_collision:
            self.robot_collision = True

        hopper = step.objects["tipping_hopper"]
        deg = abs(float(hopper.get("joint_deg", hopper.get("tilt_deg", 0.0))))
        panda_c = prefix_contacts_body(step.contacts, "panda", "tipping_hopper") or prefix_contacts_body(step.contacts, "A.", "tipping_hopper")
        ur5_c = prefix_contacts_body(step.contacts, "ur5", "receiving_cup") or prefix_contacts_body(step.contacts, "B.", "receiving_cup")

        if deg > self.pour_deg:
            if panda_c:
                self.m_A1 = True
            if ur5_c:
                self.ur5_tip_contact = True
            if external_support(step.contacts, "receiving_cup", support_names=("table", "target")):
                self.cup_supported_during_pour = True
            if panda_c and ur5_c:
                self.bilateral_tip_frames += 1

        balls_in = all(step.objects.get(b, {}).get("container") == "receiving_cup" for b in self.ball_names)
        self.all_balls_in_history.append(balls_in)

    def report(self, execution_valid: bool = True) -> TrialMetrics:
        if self._last_step is None:
            return compute_trial_metrics(
                self.task_slug, self.seed,
                {"m_A1": False, "m_A2": False}, {"m_B1": False, "m_B2": False},
                {"c_1": False, "c_2": False}, execution_valid=False,
            )
        last = self._last_step
        hopper = last.objects["tipping_hopper"]
        cup = last.objects["receiving_cup"]
        hopper_deg = abs(float(hopper.get("joint_deg", hopper.get("tilt_deg", 0.0))))
        panda_g = float(last.grippers.get("panda", last.grippers.get("A.main", 0.0)))
        ur5_g = float(last.grippers.get("ur5", last.grippers.get("B.main", 0.0)))

        m_a2 = bool(hopper_deg < self.return_deg and panda_g > 0.8)
        m_b1 = bool(self.ur5_tip_contact and not self.cup_supported_during_pour)
        cup_at_target = "cup_target_zone" in cup.get("in_regions", ())
        cup_upright = abs(float(cup.get("tilt_deg", 0.0))) < 10.0
        m_b2 = bool(cup_at_target and is_slow(cup) and cup_upright and ur5_g > 0.8)
        c1 = bool(
            self.bilateral_tip_frames >= self.min_bilateral_frames
            and not self.cup_supported_during_pour
            and not self.robot_collision
        )
        c2 = bool(
            len(self.all_balls_in_history) >= self.terminal_frames
            and all(self.all_balls_in_history[-self.terminal_frames :])
        )
        return compute_trial_metrics(
            task=self.task_slug,
            seed=self.seed,
            role_A_flags={"m_A1": self.m_A1, "m_A2": m_a2},
            role_B_flags={"m_B1": m_b1, "m_B2": m_b2},
            collab_flags={"c_1": c1, "c_2": c2},
            execution_valid=execution_valid,
            details={"bilateral_tip_frames": self.bilateral_tip_frames},
        )
