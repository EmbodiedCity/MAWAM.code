"""Atomic BF16 safetensors export and FSDP2 resumable distributed checkpoints."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import secrets
import time
import traceback


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _dist():
    import torch.distributed as dist

    return dist if dist.is_initialized() else None


def _barrier(device) -> None:
    import torch

    dist = _dist()
    if dist is not None:
        dist.barrier()
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _rank0(fn, rank: int):
    payload = [None, None]
    if rank == 0:
        try:
            payload[1] = fn()
        except BaseException:
            payload[0] = traceback.format_exc()
    dist = _dist()
    if dist is not None:
        dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError("Checkpoint rank-0 operation failed:\n" + payload[0])
    return payload[1]


def save_training_checkpoint(
    trainer,
    step: int,
    checkpoint_root: Path,
    rank: int,
    world: int,
    device,
    experiment_tag: str = "mawam_stage2",
):
    """Save BF16 inference weights (`transformer/diffusion_pytorch_model.safetensors`) and full FSDP2 resume shards."""
    import numpy as np
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
    )
    from safetensors.torch import save_file

    checkpoint_root = Path(checkpoint_root).resolve()
    _barrier(device)
    began = time.perf_counter()
    rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None,
    }
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()

    def _prepare():
        weights = checkpoint_root / "checkpoints" / f"checkpoint_step_{step}"
        resume = checkpoint_root / "resume_states" / f"checkpoint_step_{step}"
        weights.parent.mkdir(parents=True, exist_ok=True)
        resume.parent.mkdir(parents=True, exist_ok=True)
        suffix = secrets.token_hex(6)
        wt = weights.with_name(f".{weights.name}.tmp-{suffix}")
        rt = resume.with_name(f".{resume.name}.tmp-{suffix}")
        wt.mkdir()
        rt.mkdir()
        return {"weights": str(weights), "resume": str(resume), "wt": str(wt), "rt": str(rt)}

    paths = _rank0(_prepare, rank)
    wt, rt = Path(paths["wt"]), Path(paths["rt"])

    full = get_model_state_dict(
        trainer.transformer, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )

    def _write_bf16():
        model_dir = wt / "transformer"
        model_dir.mkdir()
        tensors = {k: v.detach().to(dtype=torch.bfloat16, device="cpu").contiguous() for k, v in full.items()}
        sf_path = model_dir / "diffusion_pytorch_model.safetensors"
        save_file(tensors, str(sf_path), metadata={"format": "pt", "global_step": str(step), "experiment": experiment_tag})
        cfg = dict(trainer.transformer.config)
        cfg.pop("_name_or_path", None)
        _atomic_json(model_dir / "config.json", cfg)
        return {
            "weights_bytes": sf_path.stat().st_size,
            "weights_sha256": _sha256(sf_path),
            "config_sha256": _sha256(model_dir / "config.json"),
            "tensor_count": len(tensors),
        }

    weight_receipt = _rank0(_write_bf16, rank)
    del full
    gc.collect()
    _barrier(device)

    opts = StateDictOptions(full_state_dict=False, cpu_offload=True)
    state = {
        "model": get_model_state_dict(trainer.transformer, options=opts),
        "optimizer": get_optimizer_state_dict(trainer.transformer, trainer.optimizer, options=opts),
    }
    torch.save(
        {
            "schema": 1,
            "global_step": step,
            "rank": rank,
            "world_size": world,
            "scheduler": copy.deepcopy(trainer.lr_scheduler.state_dict()),
            "rng": rng,
            "cursor": copy.deepcopy(getattr(trainer, "training_cursor", {})),
        },
        rt / f"rank_{rank:05d}.pt",
    )
    dcp.save(state, checkpoint_id=rt / "distributed", no_dist=_dist() is None)
    del state
    gc.collect()
    _barrier(device)

    def _commit():
        receipt = {
            "state": "complete",
            "global_step": step,
            "world_size": world,
            "torch_version": str(torch.__version__),
            "created_at": time.time(),
            **weight_receipt,
        }
        _atomic_json(wt / "COMPLETE.json", receipt)
        _atomic_json(rt / "COMPLETE.json", receipt)
        wt.replace(paths["weights"])
        rt.replace(paths["resume"])
        return receipt

    receipt = _rank0(_commit, rank)
    _barrier(device)
    return receipt, time.perf_counter() - began


def load_training_checkpoint(trainer, resume_directory: Path, rank: int, world: int, device):
    """Restore FSDP2 model/optimizer/scheduler shards and return per-rank RNG state."""
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )

    directory = Path(resume_directory).resolve()
    local = torch.load(directory / f"rank_{rank:05d}.pt", map_location="cpu", weights_only=False)
    opts = StateDictOptions(full_state_dict=False, cpu_offload=False)
    state = {
        "model": get_model_state_dict(trainer.transformer, options=opts),
        "optimizer": get_optimizer_state_dict(trainer.transformer, trainer.optimizer, options=opts),
    }
    metadata = dcp.FileSystemReader(directory / "distributed").read_metadata()
    saved_opt_names = {
        path[2]
        for path in metadata.planner_data.values()
        if len(path) >= 4 and tuple(path[:2]) == ("optimizer", "state")
    }
    state["optimizer"]["state"] = {
        name: state["optimizer"]["state"][name] for name in saved_opt_names if name in state["optimizer"]["state"]
    }
    dcp.load(state, checkpoint_id=directory / "distributed", no_dist=_dist() is None)
    set_model_state_dict(trainer.transformer, state["model"], options=opts)
    for group in state["optimizer"]["param_groups"]:
        for name in group["params"]:
            state["optimizer"]["state"].setdefault(name, {})
    set_optimizer_state_dict(trainer.transformer, trainer.optimizer, state["optimizer"], options=opts)
    trainer.lr_scheduler.load_state_dict(local["scheduler"])
    trainer.step = int(local["global_step"])
    trainer.training_cursor = copy.deepcopy(local.get("cursor", {}))
    del state
    _barrier(device)
    return {"global_step": trainer.step, "rng": local["rng"]}
