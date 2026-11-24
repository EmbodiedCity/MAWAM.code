"""Stage 1 (N=1) per-robot whole-episode dataset with embodiment action/view masks.

Supports both homogeneous (ALOHA+ALOHA) and heterogeneous (Panda+ALOHA, Panda+UR5,
Franka+Cobot) teams via `manifest.json` + pre-encoded causal VAE latents and frozen
T5 text embeddings.

Reference: Paper Section 4.3 & Appendix D.1.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ma_bench.action_codec import (
    ACTION_DIM,
    ACTION_PER_FRAME,
    ActionCodec,
    embodiment_action_mask,
    relative_from_absolute,
    video_spatial_loss_mask,
)

CAMERA_KEYS: Tuple[str, str, str] = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tiled_video_latents(episode: Dict[str, Any], root: Path) -> torch.Tensor:
    """Assemble per-robot `(48, F, 24, 20)` latent canvas: wrists `[:8, :]`, head `[8:24, :]`."""
    temporal = int(episode["latent_frames"])
    cameras: List[torch.Tensor] = []
    for idx, cam in enumerate(CAMERA_KEYS):
        raw_path = Path(episode["latent_paths"][cam])
        path = raw_path if raw_path.is_absolute() else (root / raw_path)
        blob = torch.load(path, map_location="cpu", weights_only=True)
        h, w = (16, 20) if idx == 0 else (8, 10)
        lat = blob["latent"]
        if lat.ndim == 2:
            lat = lat.reshape(temporal, h, w, 48)
        cameras.append(lat)
    wrists = torch.cat(cameras[1:], dim=2)  # (F, 8, 20, 48)
    combined = torch.cat([wrists, cameras[0]], dim=1).permute(3, 0, 1, 2).contiguous()
    if tuple(combined.shape) != (48, temporal, 24, 20):
        raise ValueError(f"Expected latent shape (48, {temporal}, 24, 20), got {tuple(combined.shape)}")
    return combined


def prepare_episode_actions(
    raw_action16_wxyz: np.ndarray,
    latent_frames: int,
    embodiment: str,
    codec: ActionCodec,
    is_joint_space: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build `[30, F, 16, 1]` normalized actions and bool masks with 16-slot initial dummy block."""
    raw = np.asarray(raw_action16_wxyz, dtype=np.float32)
    required = (int(latent_frames) - 1) * ACTION_PER_FRAME
    if len(raw) <= required:
        raise ValueError(f"Episode has {len(raw)} rows, need > {required} for {latent_frames} latent frames")
    valid_raw = np.array([True] * 8 + [embodiment.lower() in ("aloha", "aloha_dual", "cobot_magic", "cobot_dual")] * 8, dtype=bool)
    if is_joint_space:
        seq_rel = raw[: required + 1].astype(np.float64)
    else:
        seq_rel = relative_from_absolute(raw[: required + 1], raw[0], valid_dims=valid_raw)
    # Initial latent frame 0 carries a 16-row dummy block; future rows start at raw[1]
    sequence = np.concatenate([np.zeros((ACTION_PER_FRAME, 16), dtype=np.float64), seq_rel[1:]], axis=0)
    norm_30, mask_30 = codec.encode_slots(sequence, embodiment=embodiment)
    actions = (
        torch.from_numpy(norm_30)
        .reshape(latent_frames, ACTION_PER_FRAME, ACTION_DIM)
        .permute(2, 0, 1)
        .unsqueeze(-1)
        .contiguous()
    )
    actions_mask = (
        torch.from_numpy(mask_30)
        .reshape(latent_frames, ACTION_PER_FRAME, ACTION_DIM)
        .permute(2, 0, 1)
        .unsqueeze(-1)
        .contiguous()
    )
    return actions, actions_mask


class Stage1Dataset(Dataset):
    """Per-robot Stage 1 dataset supporting homogeneous and 1:1 mixed heterogeneous teams."""

    def __init__(
        self,
        dataset_root: str | Path,
        codec: ActionCodec,
        cfg_prob: float = 0.1,
        cache_size: int = 32,
        is_joint_space: bool = False,
    ) -> None:
        self.root = Path(dataset_root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.episodes: List[Dict[str, Any]] = list(self.manifest["episodes"])
        self.codec = codec
        self.cfg_prob = float(cfg_prob)
        self.cache_size = int(cache_size)
        self.is_joint_space = bool(is_joint_space)
        self.cache: OrderedDict[int, Dict[str, Any]] = OrderedDict()

        texts = torch.load(self.root / "text" / "text_cache.pt", map_location="cpu", weights_only=True)
        self.prompt_embs = texts["text_emb"]
        self.prompt_masks = texts["text_mask"]
        self.empty_emb = texts["empty_text_emb"]
        self.empty_mask = texts["empty_text_mask"]

    def __len__(self) -> int:
        return len(self.episodes)

    def _load_item(self, index: int) -> Dict[str, Any]:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        ep = self.episodes[index]
        act_path = Path(ep["actions_path"])
        if not act_path.is_absolute():
            act_path = self.root / act_path
        blob = torch.load(act_path, map_location="cpu", weights_only=True)
        raw = blob["raw_action16_wxyz"].numpy()
        embodiment = str(ep["embodiment"])
        actions, actions_mask = prepare_episode_actions(
            raw, int(ep["latent_frames"]), embodiment, self.codec, is_joint_space=self.is_joint_space
        )
        latents = load_tiled_video_latents(ep, self.root)
        v_mask = torch.from_numpy(video_spatial_loss_mask(embodiment))[None, None].clone()
        prompt_idx = int(ep.get("prompt_index", ep["agent_id"]))
        item = {
            "latents": latents,
            "actions": actions,
            "actions_mask": actions_mask,
            "video_spatial_mask": v_mask,
            "agent_id": int(ep["agent_id"]),
            "embodiment": embodiment,
            "joint_episode_id": str(ep["joint_episode_id"]),
            "text_emb": self.prompt_embs[prompt_idx],
            "text_mask": self.prompt_masks[prompt_idx],
        }
        if self.cache_size > 0:
            self.cache[index] = item
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return item

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = dict(self._load_item(index))
        if self.cfg_prob > 0.0 and torch.rand(()).item() < self.cfg_prob:
            item["text_emb"] = self.empty_emb
            item["text_mask"] = self.empty_mask
        return item
