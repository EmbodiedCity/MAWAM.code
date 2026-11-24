"""MA-Bench: Closed-loop multi-robot collaboration benchmark (MAWAM Section 3.3)."""

from ma_bench.action_codec import (
    ACTION_DIM,
    ACTION_PER_FRAME,
    EMBODIMENT_SLOTS,
    ActionCodec,
    absolute_from_relative,
    embodiment_action_mask,
    pack_executed_feedback,
    relative_from_absolute,
    unpack_action_chunk,
    video_spatial_loss_mask,
)
from ma_bench.metrics import (
    TrialMetrics,
    aggregate_task_metrics,
    compute_trial_metrics,
    wilson_ci,
)
from ma_bench.registry import MABENCH_TASKS, REAL_ROBOT_TASKS, TaskInfo, get_task_info

__all__ = [
    "ACTION_DIM",
    "ACTION_PER_FRAME",
    "EMBODIMENT_SLOTS",
    "MABENCH_TASKS",
    "REAL_ROBOT_TASKS",
    "ActionCodec",
    "TaskInfo",
    "TrialMetrics",
    "absolute_from_relative",
    "aggregate_task_metrics",
    "compute_trial_metrics",
    "embodiment_action_mask",
    "get_task_info",
    "pack_executed_feedback",
    "relative_from_absolute",
    "unpack_action_chunk",
    "video_spatial_loss_mask",
    "wilson_ci",
]
