"""Task-agnostic three-stage MA-Task Harness orchestrator."""

from ma_task_harness.orchestrator.backends import (
    AgentResult,
    CallableBackend,
    ClaudeCodeBackend,
    CodexBackend,
    CodingBackend,
    OpenAICompatBackend,
    make_backend,
)
from ma_task_harness.orchestrator.driver import CoEvolutionDriver, CoEvolutionResult
from ma_task_harness.orchestrator.specifier import run_specify

__all__ = [
    "AgentResult",
    "CallableBackend",
    "ClaudeCodeBackend",
    "CodexBackend",
    "CodingBackend",
    "OpenAICompatBackend",
    "CoEvolutionDriver",
    "CoEvolutionResult",
    "make_backend",
    "run_specify",
]
