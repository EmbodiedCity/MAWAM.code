"""Compiled fused kernels for real-time MAWAM inference (Paper Section 4.4 & Appendix D.4)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


@torch.compile(dynamic=True)
def fast_adaln_gate_split(temb_float: torch.Tensor, scale_shift_table: torch.Tensor):
    t = scale_shift_table[None] + temb_float
    s_msa, sc_msa, g_msa, cs_msa, csc_msa, cg_msa = rearrange(t, "b l n c -> b n l c").chunk(6, dim=1)
    return (
        s_msa.squeeze(1),
        sc_msa.squeeze(1),
        g_msa.squeeze(1),
        cs_msa.squeeze(1),
        csc_msa.squeeze(1),
        cg_msa.squeeze(1),
    )


@torch.compile(dynamic=True)
def fast_modulate_norm_act(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (F.layer_norm(x.float(), (x.shape[-1],), eps=1e-6) * (1.0 + scale) + shift).type_as(x)


@torch.compile(dynamic=True)
def fast_residual_gate(x: torch.Tensor, out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (x.float() + out.float() * gate).type_as(x)


@torch.compile(dynamic=True)
def fast_apply_rope_pair(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q0, q1 = q.float().unflatten(-1, (-1, 2)).unbind(-1)
    q_out = (
        torch.stack([q0 * cos - q1 * sin, q0 * sin + q1 * cos], dim=-1)
        .flatten(3)
        .to(q.dtype)
        .transpose(1, 2)
    )
    k0, k1 = k.float().unflatten(-1, (-1, 2)).unbind(-1)
    k_out = (
        torch.stack([k0 * cos - k1 * sin, k0 * sin + k1 * cos], dim=-1)
        .flatten(3)
        .to(k.dtype)
        .transpose(1, 2)
    )
    return q_out, k_out
