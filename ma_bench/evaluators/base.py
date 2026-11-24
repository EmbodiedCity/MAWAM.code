"""Common physical predicates and streaming base class for MA-Bench evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from ma_bench.metrics import TrialMetrics


@dataclass
class StepEvidence:
    """One deterministic physics/control state sample fed to a task evaluator."""

    time_s: float
    dt_s: float
    objects: Dict[str, Dict[str, Any]]           # name -> {pos, lin_vel, ang_vel, tilt_deg, joint_deg, ...}
    grippers: Dict[str, float]                   # arm_key -> [0..1] (0 closed, 1 open)
    contacts: Tuple[Tuple[str, str], ...] = ()   # entity name pairs in contact
    hand_positions: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    robot_collision: bool = False
    self_collision: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def has_contact(contacts: Sequence[Tuple[str, str]], entity_a: str, entity_b: str) -> bool:
    for x, y in contacts:
        if (x == entity_a and y == entity_b) or (x == entity_b and y == entity_a):
            return True
    return False


def prefix_contacts_body(contacts: Sequence[Tuple[str, str]], prefix: str, body: str) -> bool:
    for x, y in contacts:
        if (x.startswith(prefix) and body in y) or (y.startswith(prefix) and body in x):
            return True
    return False


def any_robot_contact(contacts: Sequence[Tuple[str, str]], body: str) -> bool:
    prefixes = ("robot_a", "robot_b", "panda", "aloha", "ur5", "A.", "B.")
    for x, y in contacts:
        if body in x and any(y.startswith(p) for p in prefixes):
            return True
        if body in y and any(x.startswith(p) for p in prefixes):
            return True
    return False


def external_support(contacts: Sequence[Tuple[str, str]], body: str, support_names: Sequence[str] = ("table", "start_zone", "exchange_zone", "target_zone", "cabinet_shelf", "tray")) -> bool:
    for x, y in contacts:
        if body in x and any(s in y for s in support_names):
            return True
        if body in y and any(s in x for s in support_names):
            return True
    return False


def is_slow(obj: Mapping[str, Any], max_lin: float = 0.02, max_ang: float = 0.15) -> bool:
    lin = float(np.linalg.norm(obj.get("lin_vel", (0.0, 0.0, 0.0))))
    ang = float(np.linalg.norm(obj.get("ang_vel", (0.0, 0.0, 0.0))))
    return lin < max_lin and ang < max_ang


class BaseTaskEvaluator:
    """Streaming, deterministic, LLM-free evaluator over simulator state."""

    task_slug: str = ""

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self.steps = 0
        self.time_s = 0.0
        self.events: List[Dict[str, Any]] = []

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.steps = 0
        self.time_s = 0.0
        self.events.clear()

    def update(self, step: StepEvidence) -> None:
        raise NotImplementedError

    def report(self, execution_valid: bool = True) -> TrialMetrics:
        raise NotImplementedError

    def evaluate_sequence(self, steps: Iterable[StepEvidence], execution_valid: bool = True) -> TrialMetrics:
        for st in steps:
            self.update(st)
        return self.report(execution_valid=execution_valid)
