"""MAWAM: Multi-Agent World Action Model (Paper Section 4 & Appendix D)."""

from mawam.config import (
    InferenceConfig,
    ModelConfig,
    RealRobotConfig,
    Stage1Config,
    Stage2Config,
    load_inference_config,
    load_training_config,
)

__all__ = [
    "InferenceConfig",
    "ModelConfig",
    "RealRobotConfig",
    "Stage1Config",
    "Stage2Config",
    "load_inference_config",
    "load_training_config",
]
