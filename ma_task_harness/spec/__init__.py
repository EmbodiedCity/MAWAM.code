"""Task-agnostic TaskSpec schema, YAML loader, and MA-Bench example loader."""

from ma_task_harness.spec.schema import (
    COMPLEMENTARITY_ORDER,
    ROBOT_PRESETS,
    EmbodimentSpec,
    PredicateSpec,
    ProbeConfig,
    PromptsSpec,
    TaskLabels,
    TaskSpec,
    TaskSpecValidationError,
    examples_dir,
    load_example_specs,
    load_task_spec,
    load_task_spec_from_text,
    save_task_spec,
)

__all__ = [
    "COMPLEMENTARITY_ORDER",
    "ROBOT_PRESETS",
    "EmbodimentSpec",
    "PredicateSpec",
    "ProbeConfig",
    "PromptsSpec",
    "TaskLabels",
    "TaskSpec",
    "TaskSpecValidationError",
    "examples_dir",
    "load_example_specs",
    "load_task_spec",
    "load_task_spec_from_text",
    "save_task_spec",
]
