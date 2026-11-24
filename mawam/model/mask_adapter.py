"""Embodiment action-slot masking & single-arm video view masking (Paper §4.1, §4.3, Appendix D).

Ensures inactive channels of single-arm (8-D) or 14-D embodiments stay strictly zero
across clean actions, injected noise, noisy/conditioning inputs, flow targets, and loss.
Missing single-arm wrist views preserve their real black-image VAE latents at input
and are excluded from the video loss via `video_spatial_mask`.
"""

from __future__ import annotations

from types import MethodType

import torch

from ma_bench.action_codec import EMBODIMENT_SLOTS


def masked_action(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("action mask must be bool with the exact action shape")
    return value.masked_fill(~mask, 0)


def validate_stage1_batch(batch: dict) -> None:
    action, mask = batch["actions"], batch["actions_mask"]
    if action.ndim != 5 or action.shape[1] != 30 or action.shape[3:] != (16, 1):
        raise ValueError("actions require shape [B, 30, T, 16, 1]")
    if mask.shape != action.shape or mask.dtype != torch.bool:
        raise ValueError("actions_mask requires exact bool action shape")
    latent = batch["latents"]
    video_mask = batch["video_spatial_mask"]
    if video_mask.dtype != torch.bool or video_mask.shape != (
        latent.shape[0],
        1,
        1,
        latent.shape[-2],
        latent.shape[-1],
    ):
        raise ValueError("video_spatial_mask requires [B, 1, 1, H, W] bool")
    if not bool(video_mask.flatten(1).any(dim=1).all()):
        raise ValueError("Every sample requires a nonempty video loss mask")


@torch.no_grad()
def _convert_input_format(self, batch: dict) -> dict:
    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    validate_stage1_batch(batch)
    batch["actions"] = masked_action(batch["actions"], batch["actions_mask"])
    return batch


@torch.no_grad()
def _add_noise(self, latent, train_scheduler, action_mask=None, action_mode=False, noisy_cond_prob=0.0):
    if not action_mode:
        return self._mawam_orig_add_noise(
            latent,
            train_scheduler,
            action_mask=action_mask,
            action_mode=False,
            noisy_cond_prob=noisy_cond_prob,
        )
    if action_mask is None:
        raise ValueError("Action noising requires an explicit embodiment action_mask")
    latent = masked_action(latent, action_mask)
    bindings = self._mawam_author_bindings
    batch, channels, frames, height, width = latent.shape
    timestep_ids = bindings.sample_timestep_id(
        batch_size=frames, num_train_timesteps=train_scheduler.num_train_timesteps
    )
    noise = masked_action(torch.zeros_like(latent).normal_(), action_mask)
    timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
    noisy = masked_action(train_scheduler.add_noise(latent, noise, timesteps, t_dim=2), action_mask)
    targets = masked_action(train_scheduler.training_target(latent, noise, timesteps), action_mask)
    grid = (
        bindings.get_mesh_id(frames, height, width, t=1, f_w=1, f_shift=0, action=True)
        .to(self.device)[None]
        .repeat(batch, 1, 1)
    )
    if torch.rand(1).item() < noisy_cond_prob:
        ids = bindings.sample_timestep_id(
            batch_size=frames,
            min_timestep_bd=0.5,
            max_timestep_bd=1.0,
            num_train_timesteps=train_scheduler.num_train_timesteps,
        )
        cond_noise = masked_action(torch.zeros_like(latent).normal_(), action_mask)
        cond_timesteps = train_scheduler.timesteps[ids].to(self.device)
        latent = masked_action(
            train_scheduler.add_noise(latent, cond_noise, cond_timesteps, t_dim=2), action_mask
        )
    else:
        cond_timesteps = torch.zeros_like(timesteps)
    return {
        "timesteps": timesteps[None].repeat(batch, 1),
        "noisy_latents": noisy,
        "targets": targets,
        "latent": latent,
        "cond_timesteps": cond_timesteps[None].repeat(batch, 1),
        "grid_id": grid,
    }


def guard_action_inputs(inputs: dict) -> dict:
    action = inputs["action_dict"]
    mask = action["actions_mask"]
    for key in ("latent", "noisy_latents", "targets"):
        action[key] = masked_action(action[key], mask)
    return inputs


@torch.no_grad()
def _prepare_input_dict(self, batch: dict) -> dict:
    inputs = self._mawam_orig_prepare_input_dict(batch)
    inputs["latent_dict"]["video_spatial_mask"] = batch["video_spatial_mask"]
    return guard_action_inputs(inputs)


def masked_mse_per_sample_frame(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, frame_weight: torch.Tensor
) -> torch.Tensor:
    """Mean over effective elements per sample/frame, weighted by flow schedule, then mean."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("predictions and targets must share [B, C, T, H, W] shape")
    if mask.dtype != torch.bool or frame_weight.shape != (prediction.shape[0], prediction.shape[2]):
        raise ValueError("loss masks must be bool and weights must be [B, T]")
    mask = torch.broadcast_to(mask, prediction.shape)
    count = mask.sum(dim=(1, 3, 4)).to(torch.float32)
    if not bool((count > 0).all()):
        raise ValueError("Cannot normalize loss over an empty sample/frame mask")
    residual = (prediction.float() - target.detach().float()).masked_fill(~mask, 0)
    per_frame = residual.square().sum(dim=(1, 3, 4)) / count
    return (per_frame * frame_weight.float()).mean(dim=1).mean()


def _compute_loss(self, inputs: dict, pred):
    latent_pred, action_pred = pred
    latent_target, action_target = inputs["latent_dict"]["targets"], inputs["action_dict"]["targets"]
    batch, channels, frames, slots, width = action_target.shape
    action_pred = action_pred.reshape(batch, frames, slots, channels).permute(0, 3, 1, 2).unsqueeze(-1)
    latent_pred = self._mawam_author_bindings.data_seq_to_patch(
        self.patch_size,
        latent_pred,
        latent_target.shape[-3],
        latent_target.shape[-2],
        latent_target.shape[-1],
        batch_size=latent_pred.shape[0],
    )
    vt, at = inputs["latent_dict"]["timesteps"], inputs["action_dict"]["timesteps"]
    video_weight = self.train_scheduler_latent.training_weight(vt.flatten()).reshape(vt.shape)
    action_weight = self.train_scheduler_action.training_weight(at.flatten()).reshape(at.shape)
    video = masked_mse_per_sample_frame(
        latent_pred, latent_target, inputs["latent_dict"]["video_spatial_mask"], video_weight
    )
    action = masked_mse_per_sample_frame(
        action_pred, action_target, inputs["action_dict"]["actions_mask"], action_weight
    )
    return video / self.gradient_accumulation_steps, action / self.gradient_accumulation_steps


def install_stage1_mask_adapter(trainer, upstream_train_module) -> dict:
    """Attach embodiment action/view masks onto an upstream LingBot-VA Stage 1 Trainer."""
    if hasattr(trainer, "_mawam_orig_add_noise"):
        raise RuntimeError("MAWAM Stage 1 mask adapter is already installed")
    trainer._mawam_author_bindings = upstream_train_module
    trainer._mawam_orig_add_noise = trainer._add_noise
    trainer._mawam_orig_prepare_input_dict = trainer._prepare_input_dict
    trainer.convert_input_format = MethodType(_convert_input_format, trainer)
    trainer._add_noise = MethodType(_add_noise, trainer)
    trainer._prepare_input_dict = MethodType(_prepare_input_dict, trainer)
    trainer.compute_loss = MethodType(_compute_loss, trainer)
    return {
        "schema": "mawam.stage1.embodiment_mask.v1",
        "embodiment_slots": {k: list(v) for k, v in EMBODIMENT_SLOTS.items()},
        "action_mask_sites": ["clean", "noise", "noisy", "flow_target", "conditioning", "loss"],
        "video_spatial_mask_sites": ["loss_only"],
    }
