#!/usr/bin/env python3
"""Closed-loop synchronous MA-Bench evaluation runner (100 independent trials/task).

Implements the environment side of Algorithm 2 (Paper Section 4.4 & Appendix D):
  - Connects via WebSocket to `mawam.inference.policy_server`
  - Pauses simulation physics during policy inference (synchronous benchmark protocol)
  - Executes 16 control steps on chunk 0 and 32 control steps (`C=2` latent frames)
    on every subsequent chunk at 25 Hz (250 Hz physics substeps)
  - Submits stride-4 RGB frames (6.25 Hz) and executed relative actions back to the
    policy server to ground the sliding joint KV cache (`W=64`)
  - Computes Role Score S_k, Collaboration Score S_collab, joint SR, and 95% Wilson CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ma_bench.action_codec import (
    absolute_from_relative,
    pack_executed_feedback,
    relative_from_absolute,
    unpack_action_chunk,
)
from ma_bench.envs.task_builder import build_task_env
from ma_bench.metrics import TrialMetrics, aggregate_task_metrics
from ma_bench.registry import get_task_info


class WebsocketJointClient:
    """Minimal msgpack-numpy WebSocket client talking to `mawam.inference.policy_server`."""

    def __init__(self, host: str = "127.0.0.1", port: int = 25300) -> None:
        self.url = f"ws://{host}:{port}"
        try:
            import msgpack
            import msgpack_numpy
            from websockets.sync.client import connect
        except ImportError as exc:
            raise ImportError(
                "Closed-loop evaluation requires `websockets`, `msgpack`, and `msgpack-numpy`."
            ) from exc
        self._msgpack = msgpack
        self._msgpack_numpy = msgpack_numpy
        self._ws = connect(self.url, max_size=None)

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        packed = self._msgpack.packb(payload, default=self._msgpack_numpy.encode)
        self._ws.send(packed)
        raw = self._ws.recv()
        return self._msgpack.unpackb(raw, object_hook=self._msgpack_numpy.decode, raw=False)

    def close(self) -> None:
        self._ws.close()


def run_single_trial(
    task: str,
    seed: int,
    client: WebsocketJointClient,
    backend: str = "sapien",
) -> TrialMetrics:
    info = get_task_info(task)
    env = build_task_env(info.slug, seed=seed, backend=backend)
    obs = env.reset(seed=seed)
    ref_pose = env.reference_pose_wxyz
    valid_masks = [
        np.array([True] * 8 + [r.arms == 2] * 8, dtype=bool)
        for r in info.robots
    ]

    client.infer(
        {
            "reset": True,
            "task": info.slug,
            "global_prompt": info.global_prompt,
            "local_prompts": list(info.local_prompts),
            "agent_ids": [r.agent_id for r in info.robots],
        }
    )

    max_chunks = max(1, int(np.ceil(info.nominal_duration_s / 1.28)) + 2)
    done = False
    for chunk_idx in range(max_chunks):
        first_chunk = chunk_idx == 0
        req: Dict[str, Any] = {"obs": [obs.images]} if first_chunk else {}
        resp = client.infer(req)
        rel_actions = unpack_action_chunk(resp["action"], first_chunk=first_chunk)  # [N, 16|32, 16]

        executed_rel = np.zeros_like(rel_actions)
        captured_obs: List[Dict[str, np.ndarray]] = []
        n_steps = rel_actions.shape[1]

        for t in range(n_steps):
            abs_cmd = np.zeros((info.num_agents, 16), dtype=np.float64)
            for k in range(info.num_agents):
                abs_cmd[k] = absolute_from_relative(rel_actions[k, t], ref_pose[k], valid_masks[k])
            obs, done = env.step_control(abs_cmd)
            for k in range(info.num_agents):
                executed_rel[k, t] = relative_from_absolute(obs.canonical_state[k], ref_pose[k], valid_masks[k])
            # Stride-4 video ingestion (25 Hz -> 6.25 Hz): 4 frames on chunk 0, 8 frames thereafter
            if (t + 1) % 4 == 0:
                captured_obs.append(obs.images)

        # Ground sliding joint KV cache with actual observations & executed actions
        feedback_state = pack_executed_feedback(executed_rel, first_chunk=first_chunk)
        client.infer(
            {
                "compute_kv_cache": True,
                "obs": captured_obs,
                "state": feedback_state,
            }
        )
        if done:
            break

    metrics = env.evaluator.report(execution_valid=True)
    env.close()
    return metrics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, help="MA-Bench task slug or alias (e.g. door, bowl, tray)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=25300)
    p.add_argument("--num-trials", type=int, default=100, help="Independent trajectory trials (paper: 100)")
    p.add_argument("--seed-start", type=int, default=100000)
    p.add_argument("--backend", choices=("sapien", "dry_run"), default="sapien")
    p.add_argument("--output", type=Path, default=Path("runs/eval_results.json"))
    args = p.parse_args()

    info = get_task_info(args.task)
    client = WebsocketJointClient(host=args.host, port=args.port)
    trials: List[TrialMetrics] = []
    try:
        for i in range(args.num_trials):
            seed = args.seed_start + i
            tm = run_single_trial(info.slug, seed=seed, client=client, backend=args.backend)
            trials.append(tm)
            print(
                f"[{i+1:03d}/{args.num_trials}] seed={seed} "
                f"S_role={tm.S_role*100:.1f}% S_collab={tm.S_collab*100:.1f}% SR={int(tm.joint_success)}",
                flush=True,
            )
    finally:
        client.close()

    summary = {"task": info.slug, "aggregate": aggregate_task_metrics(trials), "trials": [t.to_dict() for t in trials]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
