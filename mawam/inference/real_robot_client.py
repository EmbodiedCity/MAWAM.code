"""30 Hz Real-Robot Closed-Loop Interface (Franka FR3 8-D + AgileX Cobot Magic 14-D).

Matches Paper Section 5.5 & Appendix G:
  - Agent A: 7-DoF Franka FR3 (8 active channels: 7 joints + gripper), 3 RGB views
  - Agent B: Dual-arm AgileX Cobot Magic (14 active channels: 2x6-DoF PiPER + 2 grippers), 3 RGB views
  - Control frequency: 30 Hz; RGB ingestion: 7.5 Hz (stride 4 -> C=2 chunk = 32 steps = 1.07 s)
  - Real-time 4/8 inference preset runs in 0.72 s/chunk median on a single GPU (< 1.07 s budget),
    enabling pause-free asynchronous closed-loop hardware execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ma_bench.action_codec import EMBODIMENT_SLOTS
from mawam.config import RealRobotConfig


@dataclass
class RealRobotPacket:
    """Synchronized hardware observation + feedback packet for one real-robot chunk."""

    seq: int
    # Each agent provides 1 obs set on seq==0, 4 on seq==1, 8 on seq>=2 (7.5 Hz)
    franka_images: List[Dict[str, np.ndarray]]
    cobot_images: List[Dict[str, np.ndarray]]
    # Executed joint commands from previous chunk: 0 rows on seq==0, 16 on seq==1, 32 on seq>=2
    franka_feedback_8d: np.ndarray
    cobot_feedback_14d: np.ndarray


def validate_real_packet(packet: RealRobotPacket, cfg: RealRobotConfig | None = None) -> None:
    """Verify real-robot 30 Hz / 7.5 Hz frame counts, channel dimensions, and gripper bounds."""
    cfg = cfg or RealRobotConfig()
    expected_obs = 1 if packet.seq == 0 else (4 if packet.seq == 1 else 8)
    expected_act = 0 if packet.seq == 0 else (16 if packet.seq == 1 else 32)
    if len(packet.franka_images) != expected_obs or len(packet.cobot_images) != expected_obs:
        raise ValueError(f"seq={packet.seq}: expected {expected_obs} observation frames per robot")
    if packet.franka_feedback_8d.shape != (expected_act, len(EMBODIMENT_SLOTS["franka_fr3"])):
        raise ValueError(f"Expected Franka feedback shape ({expected_act}, 8), got {packet.franka_feedback_8d.shape}")
    if packet.cobot_feedback_14d.shape != (expected_act, len(EMBODIMENT_SLOTS["cobot_magic"])):
        raise ValueError(f"Expected Cobot feedback shape ({expected_act}, 14), got {packet.cobot_feedback_14d.shape}")
    if expected_act > 0:
        if not (np.isfinite(packet.franka_feedback_8d).all() and np.isfinite(packet.cobot_feedback_14d).all()):
            raise ValueError("Nonfinite feedback joint values")
        if np.any(packet.franka_feedback_8d[:, 7] < -1e-4) or np.any(packet.franka_feedback_8d[:, 7] > 1.0001):
            raise ValueError("Franka gripper feedback out of [0, 1]")


def pack_real_actions_to_30d(franka_8d: np.ndarray, cobot_14d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pack native Franka (T, 8) and Cobot (T, 14) joint trajectories into shared 30-D slots."""
    f = np.asarray(franka_8d, dtype=np.float32)
    c = np.asarray(cobot_14d, dtype=np.float32)
    if f.ndim != 2 or f.shape[1] != 8 or c.ndim != 2 or c.shape[1] != 14 or f.shape[0] != c.shape[0]:
        raise ValueError(f"Incompatible shapes: franka={f.shape}, cobot={c.shape}")
    t = f.shape[0]
    slots = np.zeros((2, t, 30), dtype=np.float32)
    masks = np.zeros((2, t, 30), dtype=bool)
    f_idx = list(EMBODIMENT_SLOTS["franka_fr3"])
    c_idx = list(EMBODIMENT_SLOTS["cobot_magic"])
    slots[0, :, f_idx] = f.T
    masks[0, :, f_idx] = True
    slots[1, :, c_idx] = c.T
    masks[1, :, c_idx] = True
    return slots, masks


def unpack_30d_to_real_actions(slots_30d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract native Franka (T, 8) and Cobot (T, 14) commands from `[2, T, 30]` model output."""
    arr = np.asarray(slots_30d, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 2 or arr.shape[2] != 30:
        raise ValueError(f"Expected [2, T, 30], got {arr.shape}")
    franka_8d = arr[0][:, list(EMBODIMENT_SLOTS["franka_fr3"])]
    cobot_14d = arr[1][:, list(EMBODIMENT_SLOTS["cobot_magic"])]
    return franka_8d, cobot_14d
