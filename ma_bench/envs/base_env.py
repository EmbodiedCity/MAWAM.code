"""Shared multi-robot RoboTwin 2.0 / SAPIEN environment interface (250 Hz / 25 Hz).

All paths are resolved via environment variables (`ROBOTWIN_ROOT`, `MA_BENCH_ASSETS`),
never hardcoded to private hosts or local directories.

Control contract (Paper Section 4.1 & Appendix C.1):
  - Physics runs at 250 Hz (`sub_dt = 0.004 s`, 10 substeps per control step).
  - Control commands arrive at 25 Hz (`dt = 0.04 s`) as native end-effector poses
    in each robot's canonical frame (8-D for single-arm Panda/UR5; 16-D for dual-arm ALOHA).
  - Each robot renders three RGB views (head + left_wrist + right_wrist); single-arm
    robots fill the missing right-wrist slot with zeros and mask it in training loss.
  - Synchronous evaluation pauses physics while the policy server computes each chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ma_bench.evaluators.base import BaseTaskEvaluator, StepEvidence
from ma_bench.registry import TaskInfo


@dataclass
class MultiRobotObservation:
    """Synchronized multi-view observation packet at 25 Hz (N robots x 3 RGB cameras)."""

    step: int
    time_s: float
    # keys: "observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"
    # each array has shape [N, 240, 320, 3], dtype uint8
    images: Dict[str, np.ndarray]
    # shape [N, 16] canonical measured EE poses (wxyz) + gripper states (inactive channels zero)
    canonical_state: np.ndarray


class MultiRobotEnv:
    """Base multi-robot simulation wrapper pairing a SAPIEN/RoboTwin scene with a deterministic evaluator."""

    CAMERA_KEYS: Tuple[str, str, str] = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )

    def __init__(
        self,
        task_info: TaskInfo,
        evaluator: BaseTaskEvaluator,
        robotwin_root: Optional[Path] = None,
        assets_root: Optional[Path] = None,
        image_hw: Tuple[int, int] = (240, 320),
    ) -> None:
        self.task_info = task_info
        self.evaluator = evaluator
        self.robotwin_root = Path(
            robotwin_root or os.environ.get("ROBOTWIN_ROOT", "./third_party/RoboTwin")
        )
        self.assets_root = Path(
            assets_root or os.environ.get("MA_BENCH_ASSETS", "./assets/ma_bench")
        )
        self.image_hw = image_hw
        self.control_dt = 1.0 / task_info.control_hz
        self.physics_substeps = max(1, int(round(task_info.physics_hz / task_info.control_hz)))
        self.substep_dt = self.control_dt / self.physics_substeps
        self.step_count = 0
        self.time_s = 0.0
        self._reference_pose_wxyz: Optional[np.ndarray] = None

    @property
    def num_agents(self) -> int:
        return len(self.task_info.robots)

    @property
    def reference_pose_wxyz(self) -> np.ndarray:
        if self._reference_pose_wxyz is None:
            raise RuntimeError("Call reset(seed) before accessing reference_pose_wxyz")
        return self._reference_pose_wxyz

    def reset(self, seed: int) -> MultiRobotObservation:
        raise NotImplementedError

    def step_control(self, absolute_ee_targets_wxyz: np.ndarray) -> Tuple[MultiRobotObservation, bool]:
        """Advance 1 control step (10 physics substeps @ 250 Hz), updating `self.evaluator` every substep."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class DryRunMultiRobotEnv(MultiRobotEnv):
    """Lightweight CPU dry-run environment for verifying the closed-loop client/codec without SAPIEN."""

    def __init__(self, task_info: TaskInfo, evaluator: BaseTaskEvaluator) -> None:
        super().__init__(task_info=task_info, evaluator=evaluator)
        self._state = np.zeros((self.num_agents, 16), dtype=np.float64)

    def reset(self, seed: int) -> MultiRobotObservation:
        self.step_count = 0
        self.time_s = 0.0
        self.evaluator.reset(seed=seed)
        self._state = np.zeros((self.num_agents, 16), dtype=np.float64)
        for i, r in enumerate(self.task_info.robots):
            self._state[i, 3] = 1.0  # identity quat wxyz
            self._state[i, 7] = 1.0  # open gripper
            if r.arms == 2:
                self._state[i, 11] = 1.0
                self._state[i, 15] = 1.0
        self._reference_pose_wxyz = self._state.copy()
        return self._make_obs()

    def _make_obs(self) -> MultiRobotObservation:
        h, w = self.image_hw
        images = {k: np.zeros((self.num_agents, h, w, 3), dtype=np.uint8) for k in self.CAMERA_KEYS}
        return MultiRobotObservation(
            step=self.step_count,
            time_s=self.time_s,
            images=images,
            canonical_state=self._state.copy(),
        )

    def step_control(self, absolute_ee_targets_wxyz: np.ndarray) -> Tuple[MultiRobotObservation, bool]:
        self._state = np.asarray(absolute_ee_targets_wxyz, dtype=np.float64).copy()
        self.step_count += 1
        self.time_s = self.step_count * self.control_dt
        done = self.time_s >= self.task_info.nominal_duration_s
        return self._make_obs(), done
