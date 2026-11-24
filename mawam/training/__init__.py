"""Training loops, flow-matching loss, optimizer groups, and atomic FSDP2 checkpointing."""

from mawam.training.optim import build_optimizer_groups, build_optimizer_groups_spec

__all__ = ["build_optimizer_groups", "build_optimizer_groups_spec"]
