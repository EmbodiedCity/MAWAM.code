"""Stage 2 (N>=2) synchronized paired multi-robot dataset (Paper Section 4.3 & Appendix D).

Loads full synchronized episodes (or fixed latent windows) across all robots on a shared
clock, masks the first action block from loss as conditioning history, and drops local
and global prompts jointly with probability `cfg_prob = 0.1` for CFG.
"""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from ma_bench.action_codec import ActionCodec, video_spatial_loss_mask
from mawam.data.stage1_dataset import load_tiled_video_latents, prepare_episode_actions


def window_starts(total_latent_frames: int, window_frames: int, stride: Optional[int] = None) -> List[int]:
    """Return latent frame start indices (`window_frames=0` means full episode)."""
    if window_frames <= 0 or window_frames >= total_latent_frames:
        return [0]
    step = int(stride or max(window_frames - 1, 1))
    starts = list(range(0, total_latent_frames - window_frames + 1, step))
    last = total_latent_frames - window_frames
    if starts[-1] != last:
        starts.append(last)
    return starts


class Stage2Dataset(Dataset):
    """Synchronized multi-robot paired dataset for MAWAM Stage 2 training."""

    def __init__(
        self,
        dataset_root: str | Path,
        codec: ActionCodec,
        window_frames: int = 0,
        window_stride: Optional[int] = None,
        cfg_prob: float = 0.1,
        cache_size: int = 32,
        is_joint_space: bool = False,
    ) -> None:
        self.root = Path(dataset_root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.episodes: List[Dict[str, Any]] = list(self.manifest["episodes"])
        self.num_agents = int(self.manifest.get("num_agents", len(self.episodes[0]["agents"])))
        self.codec = codec
        self.window_frames = int(window_frames)
        self.window_stride = window_stride
        self.cfg_prob = float(cfg_prob)
        self.cache_size = int(cache_size)
        self.is_joint_space = bool(is_joint_space)
        self.cache: OrderedDict[int, Dict[str, Any]] = OrderedDict()

        self.windows: List[Dict[str, Any]] = []
        for ep_idx, ep in enumerate(self.episodes):
            total_f = int(ep["latent_frames"])
            for start in window_starts(total_f, self.window_frames, self.window_stride):
                length = total_f if self.window_frames <= 0 else min(self.window_frames, total_f - start)
                self.windows.append(
                    {
                        "episode_index": ep_idx,
                        "joint_episode_id": str(ep["joint_episode_id"]),
                        "window_start_latent": start,
                        "window_frames": length,
                    }
                )

        texts = torch.load(self.root / "text" / "text_cache.pt", map_location="cpu", weights_only=True)
        self.texts = texts

    def __len__(self) -> int:
        return len(self.windows)

    def _full_episode(self, ep_idx: int) -> Dict[str, Any]:
        if ep_idx in self.cache:
            self.cache.move_to_end(ep_idx)
            return self.cache[ep_idx]
        ep = self.episodes[ep_idx]
        temporal = int(ep["latent_frames"])
        videos, actions, masks, spatial = [], [], [], []
        for agent_rec in ep["agents"]:
            src = agent_rec.get("source_stage1_episode", agent_rec)
            videos.append(load_tiled_video_latents(src, self.root))
            act_path = Path(src["actions_path"])
            if not act_path.is_absolute():
                act_path = self.root / act_path
            blob = torch.load(act_path, map_location="cpu", weights_only=True)
            raw = blob["raw_action16_wxyz"].numpy()
            embodiment = str(agent_rec.get("embodiment", src["embodiment"]))
            a, m = prepare_episode_actions(
                raw, temporal, embodiment, self.codec, is_joint_space=self.is_joint_space
            )
            actions.append(a)
            masks.append(m)
            spatial.append(torch.from_numpy(video_spatial_loss_mask(embodiment))[None, None])
        result = {
            "latents": torch.stack(videos),
            "actions": torch.stack(actions),
            "actions_mask": torch.stack(masks),
            "video_spatial_mask": torch.stack(spatial),
        }
        if self.cache_size > 0:
            self.cache[ep_idx] = result
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return result

    def __getitem__(self, index: int) -> Dict[str, Any]:
        win = self.windows[index]
        full = self._full_episode(win["episode_index"])
        start, length = win["window_start_latent"], win["window_frames"]
        item: Dict[str, Any] = {
            "latents": full["latents"][:, :, start : start + length].contiguous(),
            "actions": full["actions"][:, :, start : start + length].contiguous(),
            "actions_mask": full["actions_mask"][:, :, start : start + length].contiguous(),
            "video_spatial_mask": full["video_spatial_mask"].clone(),
            "agent_ids": torch.arange(self.num_agents, dtype=torch.int64),
            "joint_episode_id": win["joint_episode_id"],
            "window_start_latent": start,
        }
        # First action block in each window serves as conditioning history (loss-masked).
        action_loss_mask = item["actions_mask"].clone()
        action_loss_mask[:, :, 0] = False
        item["action_loss_mask"] = action_loss_mask

        if self.cfg_prob > 0.0 and torch.rand(()).item() < self.cfg_prob:
            # Jointly drop local and global prompts for CFG (Section 4.3).
            empty = self.texts["empty_text_emb"]
            empty_m = self.texts["empty_text_mask"]
            item["text_emb"] = empty.unsqueeze(0).expand(self.num_agents, -1, -1)
            item["text_mask"] = empty_m.unsqueeze(0).expand(self.num_agents, -1)
            item["global_text_emb"] = empty
            item["global_text_mask"] = empty_m
        else:
            item["text_emb"] = self.texts["text_emb"]
            item["text_mask"] = self.texts["text_mask"]
            item["global_text_emb"] = self.texts["global_text_emb"]
            item["global_text_mask"] = self.texts["global_text_mask"]
        return item
