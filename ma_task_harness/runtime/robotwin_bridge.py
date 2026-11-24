"""Bridge adapting RoboTwin 2.0 / SAPIEN state & JSONL logs into `EpisodeTrace`.

Handles native SAPIEN conventions highlighted in Paper Appendix B.3:
  - Camera extrinsics exported as either `(3, 4)` or `(4, 4)` matrices without
    false schema rejections;
  - Phase-independent physical state recovery (no reliance on expert `demo_timeline`
    phase strings);
  - Explicit separation of simulator infrastructure errors from physical violations.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ma_task_harness.runtime.state_types import (
    ArmState,
    BodyState,
    ContactEvent,
    ControlFrame,
    EpisodeTrace,
    ExecutionMode,
)


def normalize_camera_extrinsic_shape(shape: Sequence[int]) -> Tuple[int, int]:
    """Validate SAPIEN `(3, 4)` or homogeneous `(4, 4)` camera extrinsic matrix shape."""
    dims = tuple(int(x) for x in shape)
    if dims not in ((3, 4), (4, 4)):
        raise ValueError(
            f"Invalid camera extrinsic shape {dims}; expected native SAPIEN (3, 4) or homogeneous (4, 4)."
        )
    return (dims[0], dims[1])


def trace_from_jsonl_records(
    task: str,
    records: Iterable[Mapping[str, Any]],
    seed: int = 0,
    dt_s: float = 0.04,
) -> EpisodeTrace:
    """Convert recorded 25 Hz simulator step dictionaries into an `EpisodeTrace`."""
    frames: List[ControlFrame] = []
    initial_bodies: Dict[str, BodyState] = {}

    for idx, row in enumerate(records):
        t = float(row.get("time_s", idx * dt_s))
        arms: Dict[str, ArmState] = {}
        for arm_key, arm_raw in row.get("arms", {}).items():
            parts = str(arm_key).split(".", 1)
            aid = parts[0]
            aname = parts[1] if len(parts) > 1 else "main"
            arms[str(arm_key)] = ArmState(
                agent_id=aid,
                arm_name=aname,
                ee_pos=tuple(float(x) for x in arm_raw.get("ee_pos", (0.0, 0.0, 0.8))),  # type: ignore[arg-type]
                ee_quat_wxyz=tuple(
                    float(x) for x in arm_raw.get("ee_quat_wxyz", (1.0, 0.0, 0.0, 0.0))
                ),  # type: ignore[arg-type]
                gripper_open=float(arm_raw.get("gripper_open", 1.0)),
                qpos=tuple(float(x) for x in arm_raw.get("qpos", ())),
                joint_lower=tuple(float(x) for x in arm_raw.get("joint_lower", ())),
                joint_upper=tuple(float(x) for x in arm_raw.get("joint_upper", ())),
                ik_valid=bool(arm_raw.get("ik_valid", True)),
            )

        bodies: Dict[str, BodyState] = {}
        for bname, braw in row.get("bodies", {}).items():
            bs = BodyState(
                name=str(bname),
                pos=tuple(float(x) for x in braw.get("pos", (0.0, 0.0, 0.74))),  # type: ignore[arg-type]
                lin_vel=tuple(float(x) for x in braw.get("lin_vel", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
                ang_vel=tuple(float(x) for x in braw.get("ang_vel", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
                tilt_deg=float(braw.get("tilt_deg", 0.0)),
                joint_deg=float(braw.get("joint_deg", 0.0)),
                supported_by=tuple(str(s) for s in braw.get("supported_by", ("table",))),
                in_regions=tuple(str(r) for r in braw.get("in_regions", ())),
                container=str(braw["container"]) if braw.get("container") else None,
                attributes=dict(braw.get("attributes", {})),
            )
            bodies[str(bname)] = bs
            if idx == 0:
                initial_bodies[str(bname)] = bs

        contacts: List[ContactEvent] = []
        for craw in row.get("contacts", []):
            contacts.append(
                ContactEvent(
                    entity_a=str(craw["entity_a"]),
                    entity_b=str(craw["entity_b"]),
                    normal_force_n=float(craw.get("normal_force_n", 1.0)),
                    two_finger_opposed=bool(craw.get("two_finger_opposed", False)),
                    whitelisted_grasp=bool(craw.get("whitelisted_grasp", False)),
                )
            )

        cam_shapes: Dict[str, Tuple[int, int]] = {}
        for cname, cshape in row.get("camera_extrinsics_shapes", {}).items():
            cam_shapes[str(cname)] = normalize_camera_extrinsic_shape(cshape)

        frames.append(
            ControlFrame(
                step=idx,
                time_s=t,
                arms=arms,
                bodies=bodies,
                contacts=contacts,
                camera_extrinsics_shapes=cam_shapes,
                pose_overwritten=bool(row.get("pose_overwritten", False)),
                metadata=dict(row.get("metadata", {})),
            )
        )

    return EpisodeTrace(
        task=task,
        seed=seed,
        dt_s=dt_s,
        frames=frames,
        initial_bodies=initial_bodies,
        execution_mode=ExecutionMode(mode="team", seed=seed),
    )


def load_trace_jsonl(path: str | Path, task: str, seed: int = 0, dt_s: float = 0.04) -> EpisodeTrace:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    rows: List[Dict[str, Any]] = []
    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return trace_from_jsonl_records(task=task, records=rows, seed=seed, dt_s=dt_s)
