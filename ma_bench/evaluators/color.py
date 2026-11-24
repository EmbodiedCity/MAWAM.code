"""Deterministic evaluator for Task 2: Find common color (Async / Info / Homo).

Milestones (6 total):
  m_A1 / m_B1: Robot A / B lifts a block on its own side >= 25 mm.
  m_A2 / m_B2: Robot A / B places a lifted block within 25 mm of own pad and releases.
  c_1:         Both placed blocks match the unique shared color across the 2.7 m table.
  c_2:         All non-target distractor blocks move by <= 12 mm.
"""

from __future__ import annotations

from typing import Dict, Optional, Set

import numpy as np

from ma_bench.evaluators.base import BaseTaskEvaluator, StepEvidence, any_robot_contact, prefix_contacts_body
from ma_bench.metrics import TrialMetrics, compute_trial_metrics


class FindCommonColorEvaluator(BaseTaskEvaluator):
    task_slug = "find_common_color"

    def __init__(
        self,
        seed: int = 0,
        target_a: str = "block_a_shared",
        target_b: str = "block_b_shared",
        lift_min_m: float = 0.025,
        place_xy_tol_m: float = 0.025,
        untouched_max_disp_m: float = 0.012,
    ) -> None:
        super().__init__(seed=seed)
        self.target_a = target_a
        self.target_b = target_b
        self.lift_min_m = lift_min_m
        self.place_xy_tol_m = place_xy_tol_m
        self.untouched_max_disp_m = untouched_max_disp_m
        self._init_pos: Dict[str, np.ndarray] = {}
        self.lifted: Dict[str, Set[str]] = {"A": set(), "B": set()}
        self.placed: Dict[str, Set[str]] = {"A": set(), "B": set()}
        self.max_distractor_disp = 0.0
        self.bits: Dict[str, bool] = {"m_A1": False, "m_A2": False, "m_B1": False, "m_B2": False}

    def update(self, step: StepEvidence) -> None:
        self.steps += 1
        self.time_s = step.time_s
        for name, obj in step.objects.items():
            pos = np.asarray(obj["pos"], dtype=float)
            if name not in self._init_pos:
                self._init_pos[name] = pos.copy()
            side = str(obj.get("side", "A" if name.startswith("block_a") else "B"))
            prefix = ("robot_a", "A.") if side == "A" else ("robot_b", "B.")
            touched = any(prefix_contacts_body(step.contacts, p, name) for p in prefix)
            if touched and (pos[2] - self._init_pos[name][2]) >= self.lift_min_m - 1e-6:
                self.lifted[side].add(name)
                self.bits[f"m_{side}1"] = True

            pad_region = f"pad_{side}"
            on_pad = pad_region in obj.get("in_regions", ())
            if "pad_center_xy" in obj:
                on_pad = on_pad and float(np.linalg.norm(pos[:2] - np.asarray(obj["pad_center_xy"]))) <= self.place_xy_tol_m + 1e-6
            released = (not any_robot_contact(step.contacts, name)) and all(
                v > 0.8 for k, v in step.grippers.items() if k.startswith(prefix)
            )
            if name in self.lifted[side] and on_pad and released:
                self.placed[side].add(name)
                self.bits[f"m_{side}2"] = True

            if name not in (self.target_a, self.target_b):
                disp = float(np.linalg.norm(pos - self._init_pos[name]))
                self.max_distractor_disp = max(self.max_distractor_disp, disp)

    def report(self, execution_valid: bool = True) -> TrialMetrics:
        c1 = (self.target_a in self.placed["A"]) and (self.target_b in self.placed["B"])
        c2 = self.max_distractor_disp <= self.untouched_max_disp_m + 1e-9
        return compute_trial_metrics(
            task=self.task_slug,
            seed=self.seed,
            role_A_flags={"m_A1": self.bits["m_A1"], "m_A2": self.bits["m_A2"]},
            role_B_flags={"m_B1": self.bits["m_B1"], "m_B2": self.bits["m_B2"]},
            collab_flags={"c_1": c1, "c_2": c2},
            execution_valid=execution_valid,
            details={"max_distractor_disp_m": self.max_distractor_disp},
        )
