"""Deterministic evaluator for Task 4: Open door, place pot (Partial / Cap / Hetero).

Milestones (6 total):
  m_A1: Panda contacts door handle and opens spring-return cabinet door > 140 deg.
  m_A2: Panda continuously holds door > 140 deg throughout ALOHA's entire pot carry window.
  m_B1: ALOHA grasps pot bilaterally (left+right) and lifts pot >= 50 mm.
  m_B2: After m_B1, pot is inside cabinet, supported, upright, stable (< 5 cm/s), released, withdrawn.
  c_1:  m_A2 holds over a non-empty bilateral carry window with zero door-arm/pot collision.
  c_2:  All 6 placement checks hold continuously across the final 20 control frames (0.8 s)
        at the 5 cm/s terminal stability bound.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ma_bench.evaluators.base import (
    BaseTaskEvaluator,
    StepEvidence,
    any_robot_contact,
    external_support,
    prefix_contacts_body,
)
from ma_bench.metrics import TrialMetrics, compute_trial_metrics


class OpenDoorPlacePotEvaluator(BaseTaskEvaluator):
    task_slug = "open_door_place_pot"

    def __init__(
        self,
        seed: int = 0,
        door_open_deg: float = 140.0,
        pot_lift_min_m: float = 0.05,
        pot_speed_limit_mps: float = 0.05,
        terminal_frames: int = 20,
    ) -> None:
        super().__init__(seed=seed)
        self.door_open_deg = door_open_deg
        self.pot_lift_min_m = pot_lift_min_m
        self.pot_speed_limit_mps = pot_speed_limit_mps
        self.terminal_frames = terminal_frames

        self._init_pot_z: Optional[float] = None
        self.m_A1 = False
        self.m_B1 = False
        self.carry_frames = 0
        self.door_held_all_carry = True
        self.door_open_all_carry = True
        self.door_collision = False
        self.robot_collision = False
        self.placement_history: List[bool] = []

    def update(self, step: StepEvidence) -> None:
        self.steps += 1
        self.time_s = step.time_s
        if step.robot_collision:
            self.robot_collision = True

        pot = step.objects["pot"]
        door = step.objects.get("door", step.objects.get("cabinet_door", {}))
        pot_z = float(pot["pos"][2])
        if self._init_pot_z is None:
            self._init_pot_z = pot_z

        door_deg = float(door.get("joint_deg", 0.0))
        panda_door = prefix_contacts_body(step.contacts, "panda", "door") or prefix_contacts_body(step.contacts, "A.", "door")
        if panda_door and door_deg > self.door_open_deg:
            self.m_A1 = True

        aloha_left = prefix_contacts_body(step.contacts, "aloha_left", "pot") or prefix_contacts_body(step.contacts, "B.left", "pot")
        aloha_right = prefix_contacts_body(step.contacts, "aloha_right", "pot") or prefix_contacts_body(step.contacts, "B.right", "pot")
        pot_lift = pot_z - self._init_pot_z
        if aloha_left and aloha_right and pot_lift >= self.pot_lift_min_m - 1e-6:
            self.m_B1 = True

        carrying = bool(aloha_left and aloha_right and not external_support(step.contacts, "pot"))
        if carrying:
            self.carry_frames += 1
            if not panda_door:
                self.door_held_all_carry = False
            if door_deg <= self.door_open_deg:
                self.door_open_all_carry = False

        # Non-whitelisted door collision with ALOHA or pot
        for x, y in step.contacts:
            if "door" in x and (y.startswith(("aloha", "B.", "pot"))):
                self.door_collision = True
            if "door" in y and (x.startswith(("aloha", "B.", "pot"))):
                self.door_collision = True

        pot_speed = float(np.linalg.norm(pot.get("lin_vel", (0.0, 0.0, 0.0))))
        placed_ok = (
            "cabinet_interior" in pot.get("in_regions", ())
            and external_support(step.contacts, "pot", support_names=("cabinet_shelf",))
            and pot_speed < self.pot_speed_limit_mps
            and abs(float(pot.get("tilt_deg", 0.0))) < 15.0
            and not any_robot_contact(step.contacts, "pot")
            and bool(step.metadata.get("aloha_withdrawn", True))
        )
        self.placement_history.append(bool(placed_ok))

    def report(self, execution_valid: bool = True) -> TrialMetrics:
        m_a2 = bool(self.m_A1 and self.carry_frames > 0 and self.door_held_all_carry and self.door_open_all_carry)
        m_b2 = bool(self.m_B1 and any(self.placement_history))
        c1 = bool(m_a2 and not self.door_collision and not self.robot_collision)
        c2 = bool(
            c1
            and len(self.placement_history) >= self.terminal_frames
            and all(self.placement_history[-self.terminal_frames :])
        )
        return compute_trial_metrics(
            task=self.task_slug,
            seed=self.seed,
            role_A_flags={"m_A1": self.m_A1, "m_A2": m_a2},
            role_B_flags={"m_B1": self.m_B1, "m_B2": m_b2},
            collab_flags={"c_1": c1, "c_2": c2},
            execution_valid=execution_valid,
            details={"carry_frames": self.carry_frames, "door_collision": self.door_collision},
        )
