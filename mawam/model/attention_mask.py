"""Joint causal attention mask, fixed Fourier IDs, and masked role pooling (Paper §4.2, Eq. 7).

Provides both PyTorch helpers (for FlexAttention / SDPA training & inference) and
exact NumPy reference implementations so every invariant of Eq. (7) and Fig. 9
can be unit-tested on any CPU environment without GPU dependencies.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

CAUSAL_CONTRACT: Dict[str, object] = {
    "version": "stage2.shared_theta.joint_sa.v1",
    "stage": "2 * floor(frame_id / chunk_size) + modality; video=0, action=1",
    "noisy_to_clean": "key_stage < query_stage",
    "noisy_to_noisy": "key_stage == query_stage",
    "clean_to_clean": "key_stage <= query_stage",
    "clean_to_noisy": False,
    "cross_agent_sa": True,
    "pair_boundary": "no cross-episode edges",
    "local_text": "same pair and same physical agent ID",
    "global_text": "same pair, shared across all agents",
    "identity": "fixed sinusoidal physical IDs; no learned table or ID shuffle",
    "role": "masked mean of shared local text MLP outputs, RMS normalized",
    "identity_injection": "once after modality input projection, on clean/noisy video/action tokens",
    "inference_required_order": (
        "joint agent video denoising, then joint agent action denoising; "
        "replace predicted history with observations"
    ),
}


# ---------------------------------------------------------------------------
# Exact NumPy reference implementations (dependency-free CPU verification)
# ---------------------------------------------------------------------------


def fixed_player_encoding_numpy(agent_ids: np.ndarray, dim: int) -> np.ndarray:
    """Deterministic unit-RMS Fourier features phi(n_k) for nonnegative physical IDs."""
    ids = np.asarray(agent_ids, dtype=np.int64)
    if dim % 2 != 0:
        raise ValueError("Fixed player encoding requires an even hidden dimension")
    if bool((ids < 0).any()):
        raise ValueError("Physical agent IDs must be nonnegative")
    freqs = np.exp(-math.log(10000.0) * np.arange(dim // 2, dtype=np.float64) / (dim // 2))
    phase = ids.astype(np.float64)[..., None] * freqs
    encoded = np.stack((np.sin(phase), np.cos(phase)), axis=-1).reshape(*ids.shape, dim)
    return (encoded * math.sqrt(2.0)).astype(np.float32)


def masked_mean_role_numpy(projected_text: np.ndarray, text_mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMS-normalized masked mean role vector r_k over valid projected T5 tokens."""
    text = np.asarray(projected_text, dtype=np.float64)
    mask = np.asarray(text_mask, dtype=np.float64)
    if text.shape[:-1] != mask.shape:
        raise ValueError("text_mask must match projected_text token axes")
    pooled = (text * mask[..., None]).sum(axis=-2) / np.maximum(mask.sum(axis=-1, keepdims=True), 1.0)
    rms = np.sqrt(np.maximum(np.mean(pooled * pooled, axis=-1, keepdims=True), eps * eps))
    return (pooled / rms).astype(np.float32)


def build_token_metadata_numpy(
    batch_size: int,
    num_agents: int,
    latent_shape: Tuple[int, int, int],
    action_shape: Tuple[int, int, int],
    chunk_size: int,
    agent_ids: np.ndarray,
    agent_mask: Optional[np.ndarray] = None,
    frame_mask: Optional[np.ndarray] = None,
    padding: int = 0,
) -> Dict[str, np.ndarray]:
    """Token metadata for [pair, agent, noisy-video | clean-video | noisy-action | clean-action]."""
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    frames, lh, lw = map(int, latent_shape)
    af, ah, aw = map(int, action_shape)
    if af != frames:
        raise ValueError("Video and action streams must share aligned latent frame counts")
    ids = np.asarray(agent_ids, dtype=np.int64)
    if ids.ndim == 1:
        ids = np.broadcast_to(ids[None, :], (batch_size, num_agents)).copy()
    if ids.shape != (batch_size, num_agents):
        raise ValueError("agent_ids must have shape [N] or [B, N]")
    amask = np.ones((batch_size, num_agents), dtype=bool) if agent_mask is None else np.asarray(agent_mask, dtype=bool)
    fmask = np.ones((batch_size, num_agents, frames), dtype=bool) if frame_mask is None else np.asarray(frame_mask, dtype=bool)
    temporal = np.broadcast_to(np.arange(frames, dtype=np.int64)[None, None, :], (batch_size, num_agents, frames))

    vframe = np.repeat(temporal, lh * lw, axis=-1)
    aframe = np.repeat(temporal, ah * aw, axis=-1)
    vvalid = np.repeat(fmask & amask[..., None], lh * lw, axis=-1)
    avalid = np.repeat(fmask & amask[..., None], ah * aw, axis=-1)

    stage = np.concatenate(
        [
            (vframe // chunk_size) * 2,
            (vframe // chunk_size) * 2,
            (aframe // chunk_size) * 2 + 1,
            (aframe // chunk_size) * 2 + 1,
        ],
        axis=-1,
    )
    valid = np.concatenate([vvalid, vvalid, avalid, avalid], axis=-1)
    vl, al = vframe.shape[-1], aframe.shape[-1]
    clean_1d = np.concatenate(
        [
            np.zeros(vl, dtype=np.int64),
            np.ones(vl, dtype=np.int64),
            np.zeros(al, dtype=np.int64),
            np.ones(al, dtype=np.int64),
        ]
    )
    clean = np.broadcast_to(clean_1d[None, None, :], (batch_size, num_agents, stage.shape[-1]))
    pair = np.broadcast_to(np.arange(batch_size, dtype=np.int64)[:, None, None], (batch_size, num_agents, stage.shape[-1]))
    agent = np.broadcast_to(ids[..., None], (batch_size, num_agents, stage.shape[-1]))

    result = {
        "pair": pair.reshape(-1),
        "agent": agent.reshape(-1),
        "stage": stage.reshape(-1),
        "clean": clean.reshape(-1),
        "valid": valid.reshape(-1),
    }
    if padding > 0:
        for k, v in result.items():
            pad_val = False if k == "valid" else -1
            result[k] = np.pad(v, (0, padding), constant_values=pad_val)
    return result


def materialize_attention_mask_numpy(
    query_meta: Dict[str, np.ndarray],
    key_meta: Optional[Dict[str, np.ndarray]] = None,
    kind: str = "self",
    window_size: int = 64,
) -> np.ndarray:
    """Dense boolean [Q, K] realization of Eq. (7) joint causal / local / global masks."""
    km = query_meta if key_meta is None else key_meta
    if kind not in ("self", "local", "global"):
        raise ValueError(f"Unknown attention kind {kind!r}")
    valid = (
        query_meta["valid"][:, None]
        & km["valid"][None, :]
        & (query_meta["pair"][:, None] == km["pair"][None, :])
    )
    if kind == "global":
        return valid
    if kind == "local":
        return valid & (query_meta["agent"][:, None] == km["agent"][None, :])
    qs = query_meta["stage"][:, None]
    ks = km["stage"][None, :]
    qc = query_meta["clean"][:, None].astype(bool)
    kc = km["clean"][None, :].astype(bool)
    allowed = (qc & kc & (ks <= qs)) | (~qc & kc & (ks < qs)) | (~qc & ~kc & (ks == qs))
    return valid & allowed & (np.abs(qs - ks) <= window_size)


# ---------------------------------------------------------------------------
# PyTorch implementations (used by Stage2WanTransformer3DModel & JointStage2InferenceModel)
# ---------------------------------------------------------------------------


def fixed_player_encoding(agent_ids, dim: int, dtype=None):
    """PyTorch deterministic physical-ID Fourier features normalized to unit RMS."""
    import torch

    if dim % 2:
        raise ValueError("The fixed player encoding requires an even hidden dimension")
    if bool((agent_ids < 0).any()):
        raise ValueError("Physical agent IDs must be nonnegative")
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(dim // 2, device=agent_ids.device, dtype=torch.float32)
        / (dim // 2)
    )
    phase = agent_ids.float()[..., None] * frequencies
    encoded = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2) * math.sqrt(2.0)
    return encoded.to(dtype=dtype) if dtype is not None else encoded


def masked_mean_role(projected_text, text_mask, eps: float = 1e-6):
    """Pool valid projected T5 tokens into an RMS-normalized role vector r_k."""
    import torch

    if projected_text.shape[:-1] != text_mask.shape:
        raise ValueError("text_mask must match the projected text token axes")
    mask = text_mask.to(torch.float32)
    pooled = (projected_text.float() * mask[..., None]).sum(dim=-2)
    pooled = pooled / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    rms = pooled.square().mean(dim=-1, keepdim=True).clamp_min(eps * eps).sqrt()
    return (pooled / rms).to(projected_text.dtype)


def build_token_metadata(
    batch_size: int,
    num_agents: int,
    latent_shape: Sequence[int],
    action_shape: Sequence[int],
    chunk_size: int,
    agent_ids,
    agent_mask=None,
    frame_mask=None,
    temporal_frame_ids=None,
    padding: int = 0,
):
    """PyTorch metadata for [pair, agent, noisy-video/clean-video/noisy-action/clean-action]."""
    import torch
    from torch.nn import functional as F

    device = agent_ids.device
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    frames, lh, lw = map(int, latent_shape)
    af, ah, aw = map(int, action_shape)
    if af != frames:
        raise ValueError("Video/action streams must have aligned latent frame counts")
    if agent_ids.ndim == 1:
        agent_ids = agent_ids[None].expand(batch_size, -1)
    if tuple(agent_ids.shape) != (batch_size, num_agents):
        raise ValueError("agent_ids must have shape [N] or [B,N]")
    agent_mask = (
        torch.ones((batch_size, num_agents), device=device, dtype=torch.bool)
        if agent_mask is None
        else agent_mask.to(device=device, dtype=torch.bool)
    )
    frame_mask = (
        torch.ones((batch_size, num_agents, frames), device=device, dtype=torch.bool)
        if frame_mask is None
        else frame_mask.to(device=device, dtype=torch.bool)
    )
    temporal = (
        torch.arange(frames, device=device)[None, None].expand(batch_size, num_agents, -1)
        if temporal_frame_ids is None
        else temporal_frame_ids.to(device=device, dtype=torch.long)
    )
    vframe = temporal.repeat_interleave(lh * lw, dim=-1)
    aframe = temporal.repeat_interleave(ah * aw, dim=-1)
    vvalid = (frame_mask & agent_mask[..., None]).repeat_interleave(lh * lw, dim=-1)
    avalid = (frame_mask & agent_mask[..., None]).repeat_interleave(ah * aw, dim=-1)
    stage = torch.cat(
        (
            vframe // chunk_size * 2,
            vframe // chunk_size * 2,
            aframe // chunk_size * 2 + 1,
            aframe // chunk_size * 2 + 1,
        ),
        dim=-1,
    )
    valid = torch.cat((vvalid, vvalid, avalid, avalid), dim=-1)
    vl, al = vframe.shape[-1], aframe.shape[-1]
    clean = torch.cat(
        (
            torch.zeros(vl, device=device),
            torch.ones(vl, device=device),
            torch.zeros(al, device=device),
            torch.ones(al, device=device),
        )
    ).long()
    clean = clean[None, None].expand(batch_size, num_agents, -1)
    length = stage.shape[-1]
    pair = torch.arange(batch_size, device=device)[:, None, None].expand(-1, num_agents, length)
    agent = agent_ids[..., None].expand(-1, -1, length)
    result = {
        "pair": pair.flatten(),
        "agent": agent.flatten(),
        "stage": stage.flatten(),
        "clean": clean.flatten(),
        "valid": valid.flatten(),
    }
    for name, value in result.items():
        result[name] = F.pad(value, (0, padding), value=False if name == "valid" else -1)
    return result


def attention_mask_function(query_meta, key_meta=None, kind: str = "self", window_size: int = 64):
    """FlexAttention-compatible mask predicate; robot ID never gates self-attention."""
    key_meta = query_meta if key_meta is None else key_meta
    if kind not in ("self", "local", "global"):
        raise ValueError(kind)

    def predicate(batch, head, qi, ki):
        valid = (
            query_meta["valid"][qi]
            & key_meta["valid"][ki]
            & (query_meta["pair"][qi] == key_meta["pair"][ki])
        )
        if kind == "global":
            return valid
        if kind == "local":
            return valid & (query_meta["agent"][qi] == key_meta["agent"][ki])
        qs, ks = query_meta["stage"][qi], key_meta["stage"][ki]
        qc, kc = query_meta["clean"][qi].bool(), key_meta["clean"][ki].bool()
        allowed = (qc & kc & (ks <= qs)) | (~qc & kc & (ks < qs)) | (~qc & ~kc & (ks == qs))
        return valid & allowed & ((qs - ks).abs() <= window_size)

    return predicate


def materialize_attention_mask(query_meta, key_meta=None, kind: str = "self", window_size: int = 64):
    """Dense PyTorch boolean mask for SDPA inference chunks and unit tests."""
    import torch

    key_meta = query_meta if key_meta is None else key_meta
    device = query_meta["pair"].device
    q = torch.arange(query_meta["pair"].numel(), device=device)[:, None]
    k = torch.arange(key_meta["pair"].numel(), device=device)[None, :]
    return attention_mask_function(query_meta, key_meta, kind, window_size)(0, 0, q, k)
