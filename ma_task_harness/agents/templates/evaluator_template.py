"""Reference skeleton for `eval/evaluator.py` (contract shown to Evaluator E^(n))."""

from __future__ import annotations

from typing import Any, Dict, List

from ma_task_harness.runtime.state_types import EpisodeTrace
from ma_task_harness.spec.schema import TaskSpec


def check_schema(spec: TaskSpec, trace: EpisodeTrace) -> List[str]:
    """Validate non-physical trace schema contracts (e.g., SAPIEN (3,4)/(4,4) extrinsics)."""
    errors: List[str] = []
    for fr in trace.frames[:1]:
        for cam, shape in fr.camera_extrinsics_shapes.items():
            if tuple(shape) not in ((3, 4), (4, 4)):
                errors.append(f"Invalid camera extrinsic shape {shape} for {cam}; expected (3,4) or (4,4)")
    return errors


def evaluate_task(spec: TaskSpec, trace: EpisodeTrace) -> Dict[str, Any]:
    """Deterministically compute boolean flags for every predicate ID in `spec`."""
    flags: Dict[str, bool] = {}
    for pred in spec.goal:
        flags[pred.id] = False
    for preds in spec.role_milestones.values():
        for pred in preds:
            flags[pred.id] = False
    for pred in spec.collab_milestones:
        flags[pred.id] = False
    return {"flags": flags, "notes": {}}
