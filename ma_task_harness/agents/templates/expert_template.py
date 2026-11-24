"""Reference skeleton for `gen/expert.py` (contract shown to Generator G^(n))."""

from __future__ import annotations

from ma_task_harness.runtime.state_types import (
    ArmState,
    BodyState,
    ContactEvent,
    ControlFrame,
    EpisodeTrace,
    ExecutionMode,
    effective_capabilities,
)
from ma_task_harness.spec.schema import TaskSpec


def rollout_expert(spec: TaskSpec, mode: ExecutionMode) -> EpisodeTrace:
    """Synthesize and execute one episode under `mode` (`team`, `single_*`, `relax_*`, `serialize`, `delay`, `role_swap`)."""
    active_agents, caps, role_map = effective_capabilities(spec, mode)
    raise NotImplementedError(f"Implement task physics & expert planner for {spec.task}")
