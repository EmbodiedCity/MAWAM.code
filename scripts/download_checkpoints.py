#!/usr/bin/env python3
"""Download released MAWAM checkpoints from Hugging Face (`EmbodiedCity/MAWAM`)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ma_bench.registry import MABENCH_TASKS, get_task_info


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task",
        default="all",
        help="Task slug/alias (bowl, color, scan, door, tray, food, balls) or 'all'",
    )
    p.add_argument("--local-dir", type=Path, default=Path("./mawam_ckpts"))
    p.add_argument("--repo-id", default="EmbodiedCity/MAWAM")
    args = p.parse_args()

    from huggingface_hub import snapshot_download

    if args.task == "all":
        patterns = ["ckpt/**"]
    else:
        info = get_task_info(args.task)
        patterns = [f"{info.hf_checkpoint_subdir}/*"]

    local = snapshot_download(
        repo_id=args.repo_id,
        allow_patterns=patterns,
        local_dir=str(args.local_dir),
    )
    print(f"Downloaded {patterns} from {args.repo_id} -> {local}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
