"""CPU unit tests verifying Paper Section 4 & Appendix Table 6 model/training contracts."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from ma_bench.action_codec import (
    ActionCodec,
    absolute_from_relative,
    embodiment_action_mask,
    pack_executed_feedback,
    relative_from_absolute,
    unpack_action_chunk,
    video_spatial_loss_mask,
)
from mawam.config import (
    InferenceConfig,
    Stage1Config,
    Stage2Config,
    load_inference_config,
    load_training_config,
)
from mawam.inference.real_robot_client import pack_real_actions_to_30d, unpack_30d_to_real_actions
from mawam.model.attention_mask import (
    build_token_metadata_numpy,
    fixed_player_encoding_numpy,
    masked_mean_role_numpy,
    materialize_attention_mask_numpy,
)
from mawam.training.losses import masked_flow_mse_numpy
from mawam.training.optim import build_optimizer_groups_spec

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_fixed_fourier_id_unit_rms() -> None:
    ids = np.array([[0, 1, 2, 3]], dtype=np.int64)
    enc = fixed_player_encoding_numpy(ids, dim=3072)
    assert enc.shape == (1, 4, 3072)
    rms = np.sqrt(np.mean(enc.astype(np.float64) ** 2, axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-4), f"Expected unit RMS, got {rms}"
    # Distinct physical IDs produce distinct encodings
    assert not np.allclose(enc[0, 0], enc[0, 1])


def test_masked_mean_role_ignores_padding() -> None:
    rng = np.random.default_rng(0)
    proj = rng.normal(size=(2, 8, 64)).astype(np.float32)
    mask = np.array([[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]], dtype=bool)
    r1 = masked_mean_role_numpy(proj, mask)
    proj2 = proj.copy()
    proj2[~mask] = 9999.0
    r2 = masked_mean_role_numpy(proj2, mask)
    assert np.allclose(r1, r2, atol=1e-5)
    rms = np.sqrt(np.mean(r1.astype(np.float64) ** 2, axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-4)


def test_joint_causal_mask_eq7() -> None:
    """Verify Eq. (7) and Fig. 9 across two episodes, two agents, two chunks."""
    # B=2 episodes, N=2 agents, F=4 frames, chunk_size=2 -> 2 chunks (stages 0,1,2,3)
    meta = build_token_metadata_numpy(
        batch_size=2,
        num_agents=2,
        latent_shape=(4, 1, 1),
        action_shape=(4, 1, 1),
        chunk_size=2,
        agent_ids=np.array([0, 1]),
        padding=4,
    )
    sa = materialize_attention_mask_numpy(meta, kind="self", window_size=64)
    valid = meta["valid"]
    pair = meta["pair"]
    agent = meta["agent"]
    stage = meta["stage"]
    clean = meta["clean"].astype(bool)

    # 1. No cross-episode edges
    cross_ep = pair[:, None] != pair[None, :]
    assert not sa[cross_ep].any(), "Cross-episode attention leaked"

    # 2. Padding tokens never attend or get attended
    assert not sa[~valid, :].any() and not sa[:, ~valid].any()

    # 3. Clean -> Noisy strictly forbidden
    assert not sa[clean[:, None] & (~clean[None, :])].any(), "Clean->Noisy edge must be masked"

    # 4. Cross-agent noisy video (stage 0) attends to partner noisy video (stage 0) -> joint imagination!
    nv0_a = valid & (pair == 0) & (agent == 0) & (~clean) & (stage == 0)
    nv0_b = valid & (pair == 0) & (agent == 1) & (~clean) & (stage == 0)
    assert sa[np.ix_(nv0_a, nv0_b)].all(), "Noisy video must couple across agents in same stage"

    # 5. Noisy action (stage 1) attends to BOTH agents' clean video (stage 0) -> inverse dynamics readout!
    na1_a = valid & (pair == 0) & (agent == 0) & (~clean) & (stage == 1)
    cv0_b = valid & (pair == 0) & (agent == 1) & clean & (stage == 0)
    assert sa[np.ix_(na1_a, cv0_b)].all(), "Noisy action must attend to partner's clean video in same chunk"

    # 6. Local cross-attention isolates by physical agent ID; global cross-attention shares across agents
    local_meta = {
        "pair": np.array([0, 0, 1, 1]),
        "agent": np.array([0, 1, 0, 1]),
        "valid": np.array([True, True, True, True]),
    }
    global_meta = {"pair": np.array([0, 1]), "valid": np.array([True, True])}
    l_mask = materialize_attention_mask_numpy(meta, local_meta, kind="local")
    g_mask = materialize_attention_mask_numpy(meta, global_meta, kind="global")
    assert l_mask[nv0_a, 0].all() and not l_mask[nv0_a, 1].any()
    assert g_mask[nv0_a, 0].all() and g_mask[nv0_b, 0].all() and not g_mask[nv0_a, 1].any()


def test_action_codec_and_embodiment_masks() -> None:
    assert int(embodiment_action_mask("aloha").sum()) == 16
    assert int(embodiment_action_mask("panda").sum()) == 8
    assert int(embodiment_action_mask("ur5").sum()) == 8
    assert int(embodiment_action_mask("franka_fr3").sum()) == 8
    assert int(embodiment_action_mask("cobot_magic").sum()) == 14

    # Single-arm missing right-wrist video mask excludes [:8, 10:20] = 80 of 480 cells
    assert int(video_spatial_loss_mask("aloha").sum()) == 24 * 20
    assert int(video_spatial_loss_mask("panda").sum()) == 24 * 20 - 8 * 10
    assert not video_spatial_loss_mask("ur5")[:8, 10:20].any()

    # Relative <-> absolute pose roundtrip (wxyz <-> xyzw)
    ref = np.array([0.1, -0.2, 0.8, 1.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.2, 0.8, 1.0, 0.0, 0.0, 0.0, 1.0])
    ang = 0.4
    pose = ref.copy()
    pose[0:3] += np.array([0.05, -0.03, 0.07])
    pose[3:7] = np.array([np.cos(ang / 2), 0.0, np.sin(ang / 2), 0.0])  # wxyz
    pose[7] = 0.25
    rel = relative_from_absolute(pose, ref)
    rec = absolute_from_relative(rel, ref)
    assert np.allclose(rec[:3], pose[:3], atol=1e-6)
    assert np.isclose(abs(np.dot(rec[3:7], pose[3:7])), 1.0, atol=1e-6)
    assert np.isclose(rec[7], pose[7], atol=1e-6)

    # Codec roundtrip + inactive zeroing for single-arm Panda
    codec = ActionCodec(q01=np.full(30, -1.0), q99=np.full(30, 1.0), clip=5.0)
    seq = np.random.default_rng(7).uniform(-0.5, 0.5, size=(16, 16))
    norm, mask = codec.encode_slots(seq, embodiment="panda")
    assert norm.shape == (16, 30) and mask.sum() == 16 * 8
    assert np.count_nonzero(norm[~mask]) == 0
    dec = codec.decode_slots(norm, embodiment="panda")
    assert np.allclose(dec[:, :8], seq[:, :8], atol=1e-5)
    assert np.allclose(dec[:, 8:], 0.0)

    # Chunk unpack & feedback pack roundtrip (16 first chunk, 32 steady chunk)
    raw_c0 = np.random.default_rng(1).normal(size=(2, 16, 2, 16)).astype(np.float32)
    u0 = unpack_action_chunk(raw_c0, first_chunk=True)
    assert u0.shape == (2, 16, 16)
    p0 = pack_executed_feedback(u0, first_chunk=True)
    assert np.allclose(p0[:, :, 0, :], 0.0)
    assert np.allclose(unpack_action_chunk(p0, first_chunk=True), u0, atol=1e-6)

    u1 = unpack_action_chunk(raw_c0, first_chunk=False)
    assert u1.shape == (2, 32, 16)
    p1 = pack_executed_feedback(u1, first_chunk=False)
    assert np.allclose(unpack_action_chunk(p1, first_chunk=False), u1, atol=1e-6)

    # Real-robot 8-D Franka + 14-D Cobot slot roundtrip
    f8 = np.random.default_rng(2).normal(size=(32, 8)).astype(np.float32)
    c14 = np.random.default_rng(3).normal(size=(32, 14)).astype(np.float32)
    slots, rmask = pack_real_actions_to_30d(f8, c14)
    assert int(rmask[0, 0].sum()) == 8 and int(rmask[1, 0].sum()) == 14
    f8_rec, c14_rec = unpack_30d_to_real_actions(slots)
    assert np.allclose(f8_rec, f8) and np.allclose(c14_rec, c14)


def test_eq9_loss_and_table6_optimizer_groups() -> None:
    # Active-channel normalization: masked channels never dilute or affect loss
    pred = np.zeros((1, 30, 2, 16, 1), dtype=np.float32)
    target = np.ones((1, 30, 2, 16, 1), dtype=np.float32)
    mask_single = embodiment_action_mask("panda")[None, :, None, None, None]
    weights = np.ones((1, 2), dtype=np.float32)
    pred_noisy_inactive = pred.copy()
    pred_noisy_inactive[:, ~embodiment_action_mask("panda")] = 1234.0
    l1 = masked_flow_mse_numpy(pred, target, mask_single, weights)
    l2 = masked_flow_mse_numpy(pred_noisy_inactive, target, mask_single, weights)
    assert np.isclose(l1, 1.0) and np.isclose(l2, 1.0)

    # Table 6 optimizer LR groups: shared=1e-5 (wd=0.1), global=1e-5, gate=1e-4 (wd=0.0)
    groups = build_optimizer_groups_spec(
        [
            ("blocks.0.attn1.to_q.weight", 2),
            ("blocks.0.attn1.to_q.bias", 1),
            ("blocks.0.global_attn.to_q.weight", 2),
            ("blocks.0.global_gate", 1),
        ],
        lr_shared=1e-5,
        lr_global=1e-5,
        lr_gate=1e-4,
        weight_decay=0.1,
    )
    by_name = {p: (g["lr"], g["weight_decay"], g["group_name"]) for g in groups for p in g["param_names"]}
    assert by_name["blocks.0.attn1.to_q.weight"] == (1e-5, 0.1, "shared")
    assert by_name["blocks.0.attn1.to_q.bias"] == (1e-5, 0.0, "shared")
    assert by_name["blocks.0.global_attn.to_q.weight"] == (1e-5, 0.1, "global")
    assert by_name["blocks.0.global_gate"] == (1e-4, 0.0, "gate")


def test_yaml_configs_match_paper_table6() -> None:
    s1 = load_training_config(REPO_ROOT / "configs/training/stage1.yaml")
    s2 = load_training_config(REPO_ROOT / "configs/training/stage2.yaml")
    ib = load_inference_config(REPO_ROOT / "configs/inference/benchmark.yaml")
    ir = load_inference_config(REPO_ROOT / "configs/inference/realtime.yaml")

    assert isinstance(s1, Stage1Config) and s1.updates == 3000 and s1.lr == 1e-5 and s1.world_size == 8
    assert s1.betas == (0.9, 0.95) and s1.weight_decay == 0.1 and s1.grad_clip == 2.0 and s1.cfg_dropout_prob == 0.1
    assert isinstance(s2, Stage2Config) and s2.updates == 3000 and s2.lr_shared == 1e-5 and s2.lr_gate == 1e-4
    assert s2.stage2_gate_init == 1e-3 and s2.stage2_role_scale == 0.01 and s2.stage2_id_scale == 0.01

    assert isinstance(ib, InferenceConfig) and ib.chunk_frames == 2 and ib.window_frames == 64
    assert ib.video_inference_steps == 25 and ib.video_guidance_scale == 5.0 and ib.action_inference_steps == 50
    assert ir.video_inference_steps == 4 and ir.action_inference_steps == 8 and ir.use_cuda_graphs is True


def run_all() -> None:
    tests = [
        test_fixed_fourier_id_unit_rms,
        test_masked_mean_role_ignores_padding,
        test_joint_causal_mask_eq7,
        test_action_codec_and_embodiment_masks,
        test_eq9_loss_and_table6_optimizer_groups,
        test_yaml_configs_match_paper_table6,
    ]
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} model/training/config tests passed.")


if __name__ == "__main__":
    run_all()
