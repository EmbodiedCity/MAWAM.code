"""Transactional sliding joint KV cache for closed-loop MAWAM inference (Paper §4.4, Alg. 2).

Loads any released `Stage2WanTransformer3DModel` checkpoint verbatim and extends
self-attention with a per-layer sliding joint KV cache (`W = 64` latent frames).
All robots in each phase are denoised jointly before grounding with real observations.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from mawam.model.attention_mask import (
    fixed_player_encoding,
    masked_mean_role,
    materialize_attention_mask,
)
from mawam.model.stage2_transformer import Stage2Attention, Stage2WanTransformer3DModel


def _cat_metadata(parts):
    return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


def _select_metadata(meta, indices):
    return {key: value.index_select(0, indices) for key, value in meta.items()}


def _plan(query, keys, kind: str, window_size: int):
    return {
        "backend": "sdpa",
        "mask": materialize_attention_mask(query, keys, kind, window_size),
        "query_valid": query["valid"],
    }


class CachedJointAttention(Stage2Attention):
    """Self-attention backed by a transactional per-layer joint KV bank."""

    def forward(self, q, k, v, rotary_emb=None, attention_plan=None, **unused):
        if "cache_transaction" not in attention_plan:
            return super().forward(q, k, v, rotary_emb, attention_plan=attention_plan)
        transaction = attention_plan["cache_transaction"]
        query = self.norm_q(self.to_q(q)).unflatten(2, (self.heads, -1))
        key = self.norm_k(self.to_k(k)).unflatten(2, (self.heads, -1))
        value = self.to_v(v).unflatten(2, (self.heads, -1))
        if rotary_emb is not None:
            def rotate(x):
                z = torch.view_as_complex(x.double().reshape(*x.shape[:-1], -1, 2))
                return torch.view_as_real(z * rotary_emb).flatten(-2).to(x.dtype)

            query, key = rotate(query), rotate(key)
        query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        bank = transaction["bank"]
        if bank is not None:
            old = bank["layers"][self.inference_layer_index]
            indices = transaction["history_indices"]
            key = torch.cat((old["k"].index_select(2, indices), key), dim=2)
            value = torch.cat((old["v"].index_select(2, indices), value), dim=2)
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_plan["mask"], dropout_p=0.0
        )
        if transaction["persist"]:
            transaction["pending"][self.inference_layer_index] = {
                "k": key.detach(),
                "v": value.detach(),
            }
        out = out.transpose(1, 2).flatten(2, 3).to(q.dtype)
        out = self.to_out[1](self.to_out[0](out))
        return out * attention_plan["query_valid"][None, :, None].to(out.dtype)


class JointStage2InferenceModel(Stage2WanTransformer3DModel):
    """Closed-loop multi-agent WAM with sliding joint KV cache (`W=64`, `C=2`)."""

    @classmethod
    def from_training_model(cls, model):
        if not isinstance(model, Stage2WanTransformer3DModel):
            raise TypeError("Load a Stage2WanTransformer3DModel checkpoint first")
        model.__class__ = cls
        model._joint_cache_banks = {}
        model._joint_conditioning = None
        for index, block in enumerate(model.blocks):
            block.attn1.__class__ = CachedJointAttention
            block.attn1.inference_layer_index = index
        model.eval()
        return model

    @classmethod
    def from_stage2_pretrained(cls, path, **kwargs):
        device = kwargs.pop("torch_device", None)
        kwargs.pop("output_loading_info", None)
        model, info = Stage2WanTransformer3DModel.from_pretrained(
            path, output_loading_info=True, **kwargs
        )
        errors = {
            key: info.get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if info.get(key)
        }
        if errors:
            raise RuntimeError("Stage2 checkpoint must load cleanly: " + repr(errors))
        if device is not None:
            model = model.to(device)
        return cls.from_training_model(model)

    def clear_cache(self, cache_name: str = "joint"):
        self._joint_cache_banks.pop(cache_name, None)
        self._joint_conditioning = None

    def clear_pred_cache(self, cache_name: str = "joint"):
        """Evict imagined chunk tokens before writing observed ground-truth tokens (Step 4)."""
        bank = self._joint_cache_banks.get(cache_name)
        if bank is None:
            return
        indices = (~bank["meta"]["predicted"]).nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            self._joint_cache_banks.pop(cache_name, None)
            return
        layers = [
            {key: value.index_select(2, indices) for key, value in layer.items()}
            for layer in bank["layers"]
        ]
        self._joint_cache_banks[cache_name] = {
            **bank,
            "layers": layers,
            "meta": _select_metadata(bank["meta"], indices),
        }

    def cache_summary(self, cache_name: str = "joint"):
        bank = self._joint_cache_banks.get(cache_name)
        if bank is None:
            return {"cache_name": cache_name, "tokens": 0, "layers": 0}
        meta = bank["meta"]
        return {
            "cache_name": cache_name,
            "tokens": int(meta["pair"].numel()),
            "layers": len(bank["layers"]),
            "predicted_tokens": int(meta["predicted"].sum()),
            "min_frame": int(meta["frame"].min()),
            "max_frame": int(meta["frame"].max()),
            "physical_agent_ids": torch.unique(meta["agent"]).cpu().tolist(),
            "bytes": sum(
                x.numel() * x.element_size() for layer in bank["layers"] for x in layer.values()
            ),
        }

    def _conditioning(self, inputs, batch: int, agents: int, device, dtype):
        raw, mask = inputs["text_emb"], inputs["text_mask"]
        global_raw, global_mask = inputs["global_text_emb"], inputs["global_text_mask"]
        ids = torch.as_tensor(
            inputs.get("agent_ids", torch.arange(agents, device=device)),
            dtype=torch.long,
            device=device,
        )
        if ids.ndim == 1:
            ids = ids[None].expand(batch, -1)
        tensors = (raw, mask, global_raw, global_mask)
        signature = tuple(
            (x.data_ptr(), tuple(x.shape), str(x.dtype), str(x.device)) for x in tensors
        ) + (tuple(ids.flatten().cpu().tolist()), str(dtype))
        cached = self._joint_conditioning
        if cached is not None and cached["signature"] == signature:
            return cached

        local = self.condition_embedder.text_embedder(raw.to(device=device, dtype=dtype))
        global_text = self.global_text_embedder(global_raw.to(device=device, dtype=dtype))
        mask = mask.to(device=device, dtype=torch.bool)
        global_mask = global_mask.to(device=device, dtype=torch.bool)
        role = masked_mean_role(local, mask)
        injection = self.config.stage2_role_scale * role + self.config.stage2_id_scale * fixed_player_encoding(
            ids, local.shape[-1], local.dtype
        )
        local_meta = {
            "pair": torch.arange(batch, device=device)[:, None, None]
            .expand(-1, agents, local.shape[-2])
            .flatten(),
            "agent": ids[..., None].expand(-1, -1, local.shape[-2]).flatten(),
            "valid": mask.flatten(),
        }
        global_meta = {
            "pair": torch.arange(batch, device=device)[:, None]
            .expand(-1, global_text.shape[1])
            .flatten(),
            "valid": global_mask.flatten(),
        }
        cached = {
            "signature": signature,
            "raw_references": tensors,
            "ids": ids,
            "local": local.flatten(0, 2)[None],
            "global": global_text.flatten(0, 1)[None],
            "local_meta": local_meta,
            "global_meta": global_meta,
            "injection": injection,
        }
        self._joint_conditioning = cached
        return cached

    def _prepare_phase(self, inputs, action_mode: bool, clean: bool):
        value = inputs["noisy_latents"]
        batch, agents, channels, frames, height, width = value.shape
        device, dtype = value.device, self.patch_embedding_mlp.weight.dtype
        context = self._conditioning(inputs, batch, agents, device, dtype)
        hidden = self._input_embed(value.flatten(0, 1).to(dtype), "action" if action_mode else "latent")
        hidden = hidden + context["injection"].reshape(batch * agents, 1, -1)
        per_agent = hidden.shape[1]
        times = inputs["timesteps"].to(device=device)
        temb, modulation = self._time_embed(
            times.flatten(0, 1), height, width, hidden.dtype, action_mode=action_mode
        )
        grid = inputs["grid_id"].to(device=device)
        rotary = self.rope(grid.permute(2, 0, 1, 3).flatten(1)[None])[:, :, None]

        frame_ids = inputs.get("frame_ids")
        if frame_ids is None:
            frame_ids = (
                torch.arange(frames, device=device) + int(inputs.get("frame_st_id", 0))
            )[None, None].expand(batch, agents, -1)
        frame_ids = torch.as_tensor(frame_ids, dtype=torch.long, device=device)
        chunk = int(inputs.get("chunk_size", 2))
        token_frame = frame_ids.repeat_interleave(per_agent // frames, dim=-1).flatten()
        pair = torch.arange(batch, device=device)[:, None, None].expand(-1, agents, per_agent).flatten()
        physical = context["ids"][..., None].expand(-1, -1, per_agent).flatten()
        valid = torch.ones_like(pair, dtype=torch.bool)
        meta = {
            "pair": pair,
            "agent": physical,
            "frame": token_frame,
            "stage": token_frame // chunk * 2 + int(action_mode),
            "clean": torch.full_like(pair, int(clean)),
            "valid": valid,
            "predicted": torch.zeros_like(valid),
        }
        return {
            "hidden": hidden.flatten(0, 1)[None],
            "temb": temb.flatten(0, 1)[None],
            "modulation": modulation.flatten(0, 1)[None],
            "rotary": rotary,
            "meta": meta,
            "context": context,
            "shape": (batch, agents, per_agent),
            "action_mode": bool(action_mode),
        }

    def _outputs(self, hidden, phase):
        shift, scale = (self.scale_shift_table[None] + phase["temb"][:, :, None]).unbind(dim=2)
        hidden = (self.norm_out(hidden.float()) * (1 + scale) + shift).to(hidden.dtype)
        batch, agents, length = phase["shape"]
        hidden = hidden.reshape(batch, agents, length, -1)
        if phase["action_mode"]:
            return self.action_proj_out(hidden)
        return self.proj_out(hidden).reshape(batch, agents, -1, self.config.out_channels)

    @torch.no_grad()
    def inference_forward(self, inputs, action_mode: bool = False, cache_name: str = "joint", update_cache: int = 0):
        """Run one phase (`update_cache`: 0 transient, 1 predicted clean, 2 observed ground truth)."""
        if update_cache not in (0, 1, 2):
            raise ValueError("update_cache must be 0, 1, or 2")
        if update_cache and bool((inputs["timesteps"] != 0).any()):
            raise ValueError("Only denoised/observed t=0 tokens may commit to KV cache")
        phase = self._prepare_phase(inputs, action_mode, clean=(update_cache == 2))
        meta, context = phase["meta"], phase["context"]
        bank = self._joint_cache_banks.get(cache_name)
        window_frames = int(inputs.get("window_frames", 64))
        min_frame = int(meta["frame"].max()) - window_frames + 1

        if bank is not None:
            history = bank["meta"]
            history_indices = (history["frame"] >= min_frame).nonzero(as_tuple=False).flatten()
            history = _select_metadata(history, history_indices)
            if history["stage"].numel() and bool((history["stage"] >= meta["stage"].min()).any()):
                raise ValueError("Cache contains current/future stages; call clear_pred_cache() first")
            keys = _cat_metadata((history, meta))
        else:
            history_indices = torch.empty(0, dtype=torch.long, device=meta["pair"].device)
            keys = meta

        plans = {
            "self": _plan(meta, keys, "self", int(inputs.get("window_size", 64))),
            "local": _plan(meta, context["local_meta"], "local", 0),
            "global": _plan(meta, context["global_meta"], "global", 0),
        }
        transaction = {
            "bank": bank,
            "history_indices": history_indices,
            "persist": update_cache != 0,
            "pending": [None] * len(self.blocks),
        }
        plans["self"]["cache_transaction"] = transaction
        hidden = phase["hidden"]
        for block in self.blocks:
            hidden = block(
                hidden,
                context["local"],
                phase["modulation"],
                phase["rotary"],
                global_hidden_states=context["global"],
                attention_plans=plans,
            )
        output = self._outputs(hidden, phase)
        if update_cache:
            committed = {key: value.clone() for key, value in keys.items()}
            cur_len = meta["pair"].numel()
            committed["clean"][-cur_len:] = 1
            committed["predicted"][-cur_len:] = update_cache == 1
            self._joint_cache_banks[cache_name] = {
                "meta": committed,
                "layers": transaction["pending"],
                "layout": (phase["shape"][:2], tuple(context["ids"].flatten().cpu().tolist())),
            }
        return output

    def forward(self, input_dict, action_mode: bool = False, cache_name: str = "joint", update_cache: int = 0, **kwargs):
        if kwargs.get("train_mode"):
            raise RuntimeError("Use Stage2WanTransformer3DModel for training")
        return self.inference_forward(input_dict, action_mode, cache_name, update_cache)
