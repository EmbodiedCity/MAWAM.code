"""Paper-consistent configuration dataclasses for MAWAM (Table 6 & Section 4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

from ma_task_harness.spec.yaml_lite import load_yaml_file


@dataclass
class ModelConfig:
    base_model_repo: str = "robbyant/lingbot-va-base"
    model_class: str = "Stage2WanTransformer3DModel"
    architecture_version: str = "shared_theta_v1"
    num_layers: int = 30
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    in_channels: int = 48
    out_channels: int = 48
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    action_dim: int = 30
    action_per_frame: int = 16
    text_dim: int = 4096
    ffn_dim: int = 14336
    stage2_gate_init: float = 1e-3
    stage2_role_scale: float = 0.01
    stage2_id_scale: float = 0.01
    stage2_attention_backend: str = "auto"


@dataclass
class Stage1Config:
    stage: int = 1
    num_agents: int = 1
    world_size: int = 8
    episodes_per_gpu: int = 1
    updates: int = 3000
    save_interval: int = 1000
    lr: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    eps: float = 1e-8
    grad_clip: float = 2.0
    warmup_steps: int = 0
    num_train_timesteps: int = 1000
    video_snr_shift: float = 5.0
    action_snr_shift: float = 1.0
    cfg_dropout_prob: float = 0.1
    clean_video_noise_prob: float = 0.5
    chunk_size_range: Tuple[int, int] = (1, 4)
    window_size_range: Tuple[int, int] = (4, 64)
    action_clip: float = 5.0
    seed: int = 42


@dataclass
class Stage2Config:
    stage: int = 2
    num_agents: int = 2
    world_size: int = 8
    episodes_per_gpu: int = 1
    updates: int = 3000
    save_interval: int = 1000
    lr_shared: float = 1e-5
    lr_global: float = 1e-5
    lr_gate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    gate_weight_decay: float = 0.0
    eps: float = 1e-8
    grad_clip: float = 2.0
    warmup_steps: int = 0
    stage2_gate_init: float = 1e-3
    stage2_role_scale: float = 0.01
    stage2_id_scale: float = 0.01
    num_train_timesteps: int = 1000
    video_snr_shift: float = 5.0
    action_snr_shift: float = 1.0
    cfg_dropout_prob: float = 0.1
    clean_video_noise_prob: float = 0.5
    chunk_size_range: Tuple[int, int] = (1, 4)
    window_size_range: Tuple[int, int] = (4, 64)
    window_frames: int = 0  # 0 = full synchronized episode
    action_clip: float = 5.0
    seed: int = 42


@dataclass
class InferenceConfig:
    preset: str = "benchmark"
    dtype: str = "bfloat16"
    chunk_frames: int = 2
    action_per_frame: int = 16
    window_frames: int = 64
    control_hz: float = 25.0
    video_fps: float = 6.25
    video_inference_steps: int = 25
    video_guidance_scale: float = 5.0
    video_snr_shift: float = 5.0
    action_inference_steps: int = 50
    action_guidance_scale: float = 1.0
    action_snr_shift: float = 1.0
    rezero_inactive_action_every_step: bool = True
    pin_initial_observation: bool = True
    use_cuda_graphs: bool = False
    use_fused_qkv: bool = False
    use_compiled_rope: bool = False
    cache_text_kv: bool = True
    async_parallel_decoding: bool = False
    streaming_vae_cuda_graphs: bool = False

    @classmethod
    def benchmark(cls) -> "InferenceConfig":
        return cls()

    @classmethod
    def realtime(cls) -> "InferenceConfig":
        return cls(
            preset="realtime",
            control_hz=30.0,
            video_fps=7.5,
            video_inference_steps=4,
            action_inference_steps=8,
            use_cuda_graphs=True,
            use_fused_qkv=True,
            use_compiled_rope=True,
            cache_text_kv=True,
            async_parallel_decoding=True,
            streaming_vae_cuda_graphs=True,
        )


@dataclass
class RealRobotConfig:
    control_hz: float = 30.0
    video_fps: float = 7.5
    chunk_frames: int = 2
    action_per_frame: int = 16
    franka_active_channels: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 28)
    cobot_active_channels: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 28, 29)
    camera_resolutions_wh: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {"head": (320, 240), "left_wrist": (320, 240), "right_wrist": (320, 240)}
    )


def load_training_config(path: str | Path) -> Stage1Config | Stage2Config:
    raw: Dict[str, Any] = load_yaml_file(path)
    stage = int(raw.get("stage", 2))
    opt = raw.get("optimization", {})
    diff = raw.get("diffusion", {})
    data = raw.get("data", {})
    if stage == 1:
        return Stage1Config(
            world_size=int(opt.get("world_size", 8)),
            episodes_per_gpu=int(opt.get("episodes_per_gpu", 1)),
            updates=int(opt.get("updates", 3000)),
            save_interval=int(opt.get("save_interval", 1000)),
            lr=float(opt.get("lr", 1e-5)),
            betas=tuple(float(x) for x in opt.get("betas", [0.9, 0.95])),  # type: ignore[arg-type]
            weight_decay=float(opt.get("weight_decay", 0.1)),
            eps=float(opt.get("eps", 1e-8)),
            grad_clip=float(opt.get("grad_clip", 2.0)),
            warmup_steps=int(opt.get("warmup_steps", 0)),
            num_train_timesteps=int(diff.get("num_train_timesteps", 1000)),
            video_snr_shift=float(diff.get("video_snr_shift", 5.0)),
            action_snr_shift=float(diff.get("action_snr_shift", 1.0)),
            cfg_dropout_prob=float(diff.get("cfg_dropout_prob", 0.1)),
            clean_video_noise_prob=float(diff.get("clean_video_noise_prob", 0.5)),
            action_clip=float(data.get("action_clip", 5.0)),
            seed=int(data.get("seed", 42)),
        )
    arch = raw.get("architecture", {})
    return Stage2Config(
        num_agents=int(raw.get("num_agents", 2)),
        world_size=int(opt.get("world_size", 8)),
        episodes_per_gpu=int(opt.get("episodes_per_gpu", 1)),
        updates=int(opt.get("updates", 3000)),
        save_interval=int(opt.get("save_interval", 1000)),
        lr_shared=float(opt.get("lr_shared", 1e-5)),
        lr_global=float(opt.get("lr_global", 1e-5)),
        lr_gate=float(opt.get("lr_gate", 1e-4)),
        betas=tuple(float(x) for x in opt.get("betas", [0.9, 0.95])),  # type: ignore[arg-type]
        weight_decay=float(opt.get("weight_decay", 0.1)),
        gate_weight_decay=float(opt.get("gate_weight_decay", 0.0)),
        eps=float(opt.get("eps", 1e-8)),
        grad_clip=float(opt.get("grad_clip", 2.0)),
        warmup_steps=int(opt.get("warmup_steps", 0)),
        stage2_gate_init=float(arch.get("stage2_gate_init", 1e-3)),
        stage2_role_scale=float(arch.get("stage2_role_scale", 0.01)),
        stage2_id_scale=float(arch.get("stage2_id_scale", 0.01)),
        num_train_timesteps=int(diff.get("num_train_timesteps", 1000)),
        video_snr_shift=float(diff.get("video_snr_shift", 5.0)),
        action_snr_shift=float(diff.get("action_snr_shift", 1.0)),
        cfg_dropout_prob=float(diff.get("cfg_dropout_prob", 0.1)),
        clean_video_noise_prob=float(diff.get("clean_video_noise_prob", 0.5)),
        window_frames=int(data.get("window_frames", 0)),
        action_clip=float(data.get("action_clip", 5.0)),
        seed=int(data.get("seed", 42)),
    )


def load_inference_config(path: str | Path) -> InferenceConfig:
    raw: Dict[str, Any] = load_yaml_file(path)
    return InferenceConfig(
        preset=str(raw.get("preset", "benchmark")),
        dtype=str(raw.get("dtype", "bfloat16")),
        chunk_frames=int(raw.get("chunk_frames", 2)),
        action_per_frame=int(raw.get("action_per_frame", 16)),
        window_frames=int(raw.get("window_frames", 64)),
        control_hz=float(raw.get("control_hz", 25.0)),
        video_fps=float(raw.get("video_fps", 6.25)),
        video_inference_steps=int(raw.get("video_inference_steps", 25)),
        video_guidance_scale=float(raw.get("video_guidance_scale", 5.0)),
        video_snr_shift=float(raw.get("video_snr_shift", 5.0)),
        action_inference_steps=int(raw.get("action_inference_steps", 50)),
        action_guidance_scale=float(raw.get("action_guidance_scale", 1.0)),
        action_snr_shift=float(raw.get("action_snr_shift", 1.0)),
        rezero_inactive_action_every_step=bool(raw.get("rezero_inactive_action_every_step", True)),
        pin_initial_observation=bool(raw.get("pin_initial_observation", True)),
        use_cuda_graphs=bool(raw.get("use_cuda_graphs", False)),
        use_fused_qkv=bool(raw.get("use_fused_qkv", False)),
        use_compiled_rope=bool(raw.get("use_compiled_rope", False)),
        cache_text_kv=bool(raw.get("cache_text_kv", True)),
        async_parallel_decoding=bool(raw.get("async_parallel_decoding", False)),
        streaming_vae_cuda_graphs=bool(raw.get("streaming_vae_cuda_graphs", False)),
    )
