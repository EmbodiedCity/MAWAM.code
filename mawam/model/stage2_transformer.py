"""MAWAM Stage 2 Transformer: one weight-shared LingBot-VA backbone along an agent axis.

Retains exact checkpoint parameter names (`Stage2WanTransformer3DModel`, `global_attn`,
`global_norm`, `global_gate`, `global_text_embedder`) so weights released on
`https://huggingface.co/EmbodiedCity/MAWAM` load verbatim with `from_pretrained`.

Reference: Paper Section 4.2, Eq. (7)–(8), Algorithm 1, Appendix Table 6.
"""

from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F
from diffusers.configuration_utils import register_to_config

from modules.model import (
    FlexAttnFunc,
    WanAttention,
    WanTransformer3DModel,
    WanTransformerBlock,
)

from mawam.model.attention_mask import (
    CAUSAL_CONTRACT,
    attention_mask_function,
    build_token_metadata,
    fixed_player_encoding,
    masked_mean_role,
    materialize_attention_mask,
)


def _attention_plan(query_meta, key_meta, kind: str, window_size: int, backend: str):
    device = query_meta["pair"].device
    if backend == "auto":
        backend = "flex" if device.type == "cuda" else "sdpa"
    if backend == "sdpa":
        if device.type == "cuda" and query_meta["pair"].numel() > 4096:
            raise RuntimeError("Dense SDPA mask is restricted to small diagnostic sequences")
        mask = materialize_attention_mask(query_meta, key_meta, kind, window_size)
    elif backend == "flex":
        predicate = attention_mask_function(query_meta, key_meta, kind, window_size)
        mask = FlexAttnFunc.compiled_create_block_mask(
            predicate,
            1,
            1,
            query_meta["pair"].numel(),
            key_meta["pair"].numel(),
            device=device,
            _compile=True,
        )
    else:
        raise ValueError("stage2_attention_backend must be auto, flex, or sdpa")
    return {"backend": backend, "mask": mask, "query_valid": query_meta["valid"]}


class Stage2Attention(WanAttention):
    """Weight-compatible WanAttention with explicit call-local attention plans."""

    def forward(self, q, k, v, rotary_emb=None, attention_plan=None, **unused):
        if attention_plan is None:
            raise ValueError("Stage2 attention requires an explicit attention plan")
        query = self.norm_q(self.to_q(q)).unflatten(2, (self.heads, -1))
        key = self.norm_k(self.to_k(k)).unflatten(2, (self.heads, -1))
        value = self.to_v(v).unflatten(2, (self.heads, -1))
        if rotary_emb is not None:
            def rotate(x):
                complex_x = torch.view_as_complex(x.double().reshape(*x.shape[:-1], -1, 2))
                return torch.view_as_real(complex_x * rotary_emb).flatten(-2).to(x.dtype)

            query, key = rotate(query), rotate(key)
        query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        if attention_plan["backend"] == "flex":
            out = FlexAttnFunc.flex_attn(
                query,
                key,
                value,
                block_mask=attention_plan["mask"],
                kernel_options={
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_M1": 32,
                    "BLOCK_N1": 64,
                    "BLOCK_M2": 64,
                    "BLOCK_N2": 32,
                },
            )
        else:
            out = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_plan["mask"], dropout_p=0.0
            )
        out = out.transpose(1, 2).flatten(2, 3).to(q.dtype)
        out = self.to_out[1](self.to_out[0](out))
        return out * attention_plan["query_valid"][None, :, None].to(out.dtype)


class Stage2TransformerBlock(WanTransformerBlock):
    """One weight-shared MAWAM DiT block (Algorithm 1): Joint SA -> Gated Global CA -> Local CA -> FFN."""

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        global_hidden_states,
        attention_plans,
    ):
        modulation = (self.scale_shift_table[None] + temb.float()).unbind(dim=2)
        shift, scale, gate, ff_shift, ff_scale, ff_gate = modulation
        normed = (self.norm1(hidden_states.float()) * (1 + scale) + shift).to(hidden_states.dtype)
        attended = self.attn1(normed, normed, normed, rotary_emb, attention_plan=attention_plans["self"])
        hidden_states = (hidden_states.float() + attended * gate).to(hidden_states.dtype)
        normed = self.global_norm(hidden_states.float()).to(hidden_states.dtype)
        global_out = self.global_attn(
            normed, global_hidden_states, global_hidden_states, attention_plan=attention_plans["global"]
        )
        hidden_states = (hidden_states.float() + self.global_gate.float() * global_out.float()).to(
            hidden_states.dtype
        )
        normed = self.norm2(hidden_states.float()).to(hidden_states.dtype)
        hidden_states = hidden_states + self.attn2(
            normed, encoder_hidden_states, encoder_hidden_states, attention_plan=attention_plans["local"]
        )
        normed = (self.norm3(hidden_states.float()) * (1 + ff_scale) + ff_shift).to(hidden_states.dtype)
        hidden_states = (hidden_states.float() + self.ffn(normed).float() * ff_gate).to(hidden_states.dtype)
        return hidden_states * attention_plans["self"]["query_valid"][None, :, None].to(hidden_states.dtype)


class Stage2WanTransformer3DModel(WanTransformer3DModel):
    """Weight-shared multi-agent WAM transformer (`shared_theta_v1`, ~6.24B params)."""

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-6,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode="flex",
        stage2_gate_init=1e-3,
        stage2_role_scale=0.01,
        stage2_id_scale=0.01,
        stage2_attention_backend="auto",
        stage2_architecture_version="shared_theta_v1",
    ):
        super().__init__(
            patch_size=list(patch_size),
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            action_dim=action_dim,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
            attn_mode=attn_mode,
        )
        self._init_stage2(
            stage2_gate_init,
            stage2_role_scale,
            stage2_id_scale,
            stage2_attention_backend,
            stage2_architecture_version,
        )

    def _init_stage2(self, gate_init, role_scale, id_scale, backend, version):
        self.global_text_embedder = deepcopy(self.condition_embedder.text_embedder)
        for block in self.blocks:
            block.__class__ = Stage2TransformerBlock
            block.attn1.__class__ = Stage2Attention
            block.attn2.__class__ = Stage2Attention
            block.global_attn = deepcopy(block.attn2)
            block.global_norm = deepcopy(block.norm2)
            # 1-element vector retains scalar broadcasting while supporting FSDP2 sharding.
            block.global_gate = nn.Parameter(
                torch.full(
                    (1,),
                    float(gate_init),
                    dtype=block.scale_shift_table.dtype,
                    device=block.scale_shift_table.device,
                )
            )
        self.register_to_config(
            stage2_gate_init=float(gate_init),
            stage2_role_scale=float(role_scale),
            stage2_id_scale=float(id_scale),
            stage2_attention_backend=backend,
            stage2_architecture_version=version,
            _class_name=type(self).__name__,
        )

    @classmethod
    def from_stage1_model(
        cls,
        model,
        stage2_gate_init=1e-3,
        stage2_role_scale=0.01,
        stage2_id_scale=0.01,
        stage2_attention_backend="auto",
    ):
        """Promote an in-memory Stage 1 backbone to Stage 2 without duplicating weights."""
        if isinstance(model, cls) or hasattr(model, "global_text_embedder"):
            raise ValueError("Model already has Stage2 extensions")
        model.__class__ = cls
        model._init_stage2(
            stage2_gate_init, stage2_role_scale, stage2_id_scale, stage2_attention_backend, "shared_theta_v1"
        )
        return model

    @classmethod
    def from_stage1_pretrained(cls, path, **kwargs):
        stage2_args = {k: kwargs.pop(k) for k in list(kwargs) if k.startswith("stage2_")}
        kwargs.pop("output_loading_info", None)
        base, info = WanTransformer3DModel.from_pretrained(path, output_loading_info=True, **kwargs)
        failures = {
            k: info.get(k)
            for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if info.get(k)
        }
        if failures:
            raise RuntimeError("Stage1 checkpoint did not load exactly: " + repr(failures))
        return cls.from_stage1_model(base, **stage2_args)

    def architecture_config(self):
        return {
            "model_class": type(self).__name__,
            "config": dict(self.config),
            "causal_contract": deepcopy(CAUSAL_CONTRACT),
            "one_shared_backbone": True,
            "dynamic_agent_count": True,
            "learned_player_parameters": 0,
        }

    def forward(self, input_dict, train_mode=True, **kwargs):
        if not train_mode or kwargs.get("update_cache", 0):
            raise NotImplementedError(
                "Use `mawam.inference.joint_kv_model.JointStage2InferenceModel` for cached closed-loop inference."
            )
        return self.forward_train(input_dict)

    def forward_train(self, input_dict):
        vd, ad = input_dict["latent_dict"], input_dict["action_dict"]
        B, N, channels, frames, height, width = vd["noisy_latents"].shape
        if self.patch_size[0] != 1:
            raise ValueError("Stage2 temporal alignment requires patch_size[0]=1")
        if ad["noisy_latents"].shape[:2] != (B, N) or ad["noisy_latents"].shape[3] != frames:
            raise ValueError("All agents' video and action streams must be temporally aligned")
        device = vd["noisy_latents"].device
        dtype = self.patch_embedding_mlp.weight.dtype

        def embed(value, kind):
            return self._input_embed(value.flatten(0, 1).to(dtype), input_type=kind)

        noisy_v, clean_v = embed(vd["noisy_latents"], "latent"), embed(vd["latent"], "latent")
        noisy_a, clean_a = embed(ad["noisy_latents"], "action"), embed(ad["latent"], "action")

        local_raw = vd["text_emb"]
        local_mask = vd.get("text_mask", input_dict.get("local_text_mask"))
        global_mask = input_dict.get("global_text_mask")
        if local_mask is None or global_mask is None:
            raise ValueError("Explicit local text_mask and global_text_mask are required")
        local_text = self.condition_embedder.text_embedder(local_raw.to(dtype))
        global_text = self.global_text_embedder(input_dict["global_text_emb"].to(dtype))

        ids = torch.as_tensor(
            input_dict.get("agent_ids", torch.arange(N, device=device)), device=device, dtype=torch.long
        )
        if ids.ndim == 1:
            ids = ids[None].expand(B, -1)
        if tuple(ids.shape) != (B, N) or any(torch.unique(row).numel() != N for row in ids):
            raise ValueError("Physical agent IDs must be unique within each episode")

        local_mask = local_mask.to(device=device, dtype=torch.bool)
        global_mask = global_mask.to(device=device, dtype=torch.bool)
        role = masked_mean_role(local_text, local_mask)
        identity = fixed_player_encoding(ids, noisy_v.shape[-1], dtype)
        injection = self.config.stage2_role_scale * role + self.config.stage2_id_scale * identity

        splits = [noisy_v.shape[1], clean_v.shape[1], noisy_a.shape[1], clean_a.shape[1]]
        hidden = torch.cat((noisy_v, clean_v, noisy_a, clean_a), dim=1)
        hidden = hidden + injection.reshape(B * N, 1, -1)
        per_agent_length = hidden.shape[1]
        hidden = hidden.flatten(0, 1)[None]

        def time_features(stream, h, w, action):
            times = torch.cat(
                (stream["timesteps"].flatten(0, 1), stream["cond_timesteps"].flatten(0, 1)), dim=1
            )
            return self._time_embed(times, h, w, dtype=hidden.dtype, action_mode=action)

        vtemb, vmod = time_features(vd, height, width, False)
        atemb, amod = time_features(
            ad, ad["noisy_latents"].shape[-2], ad["noisy_latents"].shape[-1], True
        )
        temb = torch.cat((vtemb, atemb), dim=1).flatten(0, 1)[None]
        modulation = torch.cat((vmod, amod), dim=1).flatten(0, 1)[None]

        grid = torch.cat((vd["grid_id"], vd["grid_id"], ad["grid_id"], ad["grid_id"]), dim=-1)
        grid = grid.permute(2, 0, 1, 3).flatten(1)[None]
        rotary = self.rope(grid)[:, :, None]

        padding = (128 - hidden.shape[1] % 128) % 128
        hidden = F.pad(hidden, (0, 0, 0, padding))
        temb = F.pad(temb, (0, 0, 0, padding))
        modulation = F.pad(modulation, (0, 0, 0, 0, 0, padding))
        rotary = F.pad(rotary, (0, 0, 0, 0, 0, padding))

        query_meta = build_token_metadata(
            B,
            N,
            (frames, height // self.patch_size[1], width // self.patch_size[2]),
            ad["noisy_latents"].shape[-3:],
            input_dict["chunk_size"],
            ids,
            input_dict.get("agent_mask"),
            input_dict.get("frame_mask"),
            input_dict.get("temporal_frame_ids"),
            padding,
        )
        active_agents = input_dict.get(
            "agent_mask", torch.ones((B, N), device=device, dtype=torch.bool)
        ).to(device=device, dtype=torch.bool)
        local_tokens = local_text.shape[-2]
        local_meta = {
            "pair": torch.arange(B, device=device)[:, None, None].expand(-1, N, local_tokens).flatten(),
            "agent": ids[..., None].expand(-1, -1, local_tokens).flatten(),
            "valid": (local_mask & active_agents[..., None]).flatten(),
        }
        global_meta = {
            "pair": torch.arange(B, device=device)[:, None].expand(-1, global_text.shape[1]).flatten(),
            "valid": global_mask.flatten(),
        }

        backend = self.config.stage2_attention_backend
        plans = {
            "self": _attention_plan(query_meta, query_meta, "self", input_dict["window_size"], backend),
            "local": _attention_plan(query_meta, local_meta, "local", input_dict["window_size"], backend),
            "global": _attention_plan(query_meta, global_meta, "global", input_dict["window_size"], backend),
        }

        local_text = local_text.flatten(0, 2)[None]
        global_text = global_text.flatten(0, 1)[None]
        for block in self.blocks:
            hidden = block(
                hidden,
                local_text,
                modulation,
                rotary,
                global_hidden_states=global_text,
                attention_plans=plans,
            )

        shift, scale = (self.scale_shift_table[None] + temb[:, :, None]).unbind(dim=2)
        hidden = (self.norm_out(hidden.float()) * (1 + scale) + shift).to(hidden.dtype)
        hidden = hidden[:, : B * N * per_agent_length].reshape(B, N, per_agent_length, -1)
        video, _, action, _ = hidden.split(splits, dim=2)
        video = self.proj_out(video).reshape(B, N, -1, self.config.out_channels)
        action = self.action_proj_out(action)
        valid = query_meta["valid"][: B * N * per_agent_length].reshape(B, N, per_agent_length)
        video_valid, _, action_valid, _ = valid.split(splits, dim=2)
        video = video * video_valid.repeat_interleave(math.prod(self.patch_size), dim=2)[..., None].to(video.dtype)
        action = action * action_valid[..., None].to(action.dtype)
        return video, action
