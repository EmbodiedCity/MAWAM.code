"""MAWAM flow-matching loss (Paper Section 4.3, Eq. 9).

Combines per-agent view-masked video velocity MSE and active-channel action velocity
MSE, normalized over active slots/views per frame so single-arm embodiments are not
diluted by masked channels.
"""

from __future__ import annotations

import numpy as np


def masked_flow_mse_numpy(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray, frame_weights: np.ndarray
) -> float:
    """Exact NumPy reference for Eq. (9) per-frame effective-element masked MSE."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    m = np.broadcast_to(np.asarray(mask, dtype=bool), p.shape)
    w = np.asarray(frame_weights, dtype=np.float64)
    if p.ndim != 5 or p.shape != t.shape:
        raise ValueError("Expected [B, C, F, H, W] predictions and targets")
    if w.shape != (p.shape[0], p.shape[2]):
        raise ValueError("Expected frame_weights of shape [B, F]")
    residual = np.where(m, p - t, 0.0)
    counts = m.sum(axis=(1, 3, 4)).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError("Empty mask on at least one sample/frame")
    per_frame = (residual * residual).sum(axis=(1, 3, 4)) / counts
    return float((per_frame * w).mean())


def masked_compute_loss(trainer, input_dict, pred, video_spatial_mask, data_seq_to_patch):
    """PyTorch per-agent video + action flow-matching loss (Eq. 9)."""
    import torch
    import torch.nn.functional as F
    from einops import rearrange

    latent_pred, action_pred = pred
    latent_target = input_dict["latent_dict"]["targets"]
    action_target = input_dict["action_dict"]["targets"]
    action_pred = rearrange(action_pred, "b (f n) c -> b c f n 1", f=action_target.shape[-3])
    latent_pred = data_seq_to_patch(
        trainer.patch_size,
        latent_pred,
        latent_target.shape[-3],
        latent_target.shape[-2],
        latent_target.shape[-1],
        batch_size=latent_pred.shape[0],
    )
    bn, fn = input_dict["latent_dict"]["timesteps"].shape
    latent_w = trainer.train_scheduler_latent.training_weight(
        input_dict["latent_dict"]["timesteps"].flatten()
    ).reshape(bn, fn)
    action_w = trainer.train_scheduler_action.training_weight(
        input_dict["action_dict"]["timesteps"].flatten()
    ).reshape(bn, fn)

    if video_spatial_mask.dtype != torch.bool or video_spatial_mask.shape != (bn, 1, 1) + tuple(
        latent_target.shape[-2:]
    ):
        raise ValueError("video_spatial_mask requires [B, 1, 1, H, W] bool")
    vmask = torch.broadcast_to(video_spatial_mask, latent_target.shape).float()
    latent_loss = F.mse_loss(latent_pred.float(), latent_target.float().detach(), reduction="none")
    latent_loss = latent_loss * latent_w[:, None, :, None, None] * vmask
    latent_loss = latent_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
    vcount = vmask.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1).sum(dim=1)
    if not bool((vcount > 0).all()):
        raise ValueError("Empty video loss mask")
    latent_loss = (latent_loss.sum(dim=1) / (vcount + 1e-6)).mean()

    amask = input_dict["action_dict"]["actions_mask"].float()
    action_loss = F.mse_loss(action_pred.float(), action_target.float().detach(), reduction="none")
    action_loss = action_loss * action_w[:, None, :, None, None] * amask
    action_loss = action_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
    acount = amask.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1).sum(dim=1)
    action_loss = (action_loss.sum(dim=1) / (acount + 1e-6)).mean()

    return (
        latent_loss / trainer.gradient_accumulation_steps,
        action_loss / trainer.gradient_accumulation_steps,
    )
