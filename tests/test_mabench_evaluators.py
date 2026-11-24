"""CPU unit tests verifying all 7 MA-Bench deterministic evaluators (43 milestones) & Wilson CI."""

from __future__ import annotations

from ma_bench.evaluators import EVALUATOR_REGISTRY, StepEvidence, make_evaluator
from ma_bench.metrics import aggregate_task_metrics, wilson_ci
from ma_bench.registry import MABENCH_TASKS, REAL_ROBOT_TASKS


def test_registry_counts_match_paper() -> None:
    assert len(MABENCH_TASKS) == 7
    assert len(REAL_ROBOT_TASKS) == 3
    assert sum(t.demo_pairs for t in MABENCH_TASKS.values()) == 700
    assert sum(t.total_milestones for t in MABENCH_TASKS.values()) == 43
    assert all(t.eval_trials == 100 for t in MABENCH_TASKS.values())
    assert all(t.demo_pairs == 80 and t.eval_trials == 20 for t in REAL_ROBOT_TASKS.values())
    assert set(EVALUATOR_REGISTRY) == set(MABENCH_TASKS)


def test_wilson_ci_matches_paper_appendix() -> None:
    # Paper Table 8: 88/100 -> [80.2, 93.0], 0/100 -> [0.0, 3.7]
    lo, hi = wilson_ci(88, 100)
    assert lo == 80.2 and hi == 93.0, (lo, hi)
    lo0, hi0 = wilson_ci(0, 100)
    assert lo0 == 0.0 and hi0 == 3.7, (lo0, hi0)


def test_bowl_evaluator_pass_and_overlap_fail() -> None:
    dt = 0.04
    steps = []
    for i in range(60):
        t = i * dt
        if i < 5:
            pos, c = (-0.42, 0.0, 0.74), (("bowl", "table"),)
        elif i < 15:
            pos, c = (-0.20, 0.0, 0.80), (("robot_a_left", "bowl"),)
        elif i < 22:
            pos, c = (0.0, 0.0, 0.74), (("bowl", "table"), ("bowl", "exchange_zone"))
        elif i < 32:
            pos, c = (0.20, 0.0, 0.80), (("robot_b_left", "bowl"),)
        else:
            pos, c = (0.45, 0.0, 0.74), (("bowl", "table"), ("bowl", "target_zone"))
        steps.append(StepEvidence(time_s=t, dt_s=dt, objects={"bowl": {"pos": pos}}, grippers={}, contacts=c))

    ev = make_evaluator("bowl")
    res = ev.evaluate_sequence(steps)
    assert res.joint_success and res.S_role == 1.0 and res.S_collab == 1.0

    # Inject simultaneous grasp overlap -> c_1 fails!
    steps_bad = list(steps)
    steps_bad[10] = StepEvidence(
        time_s=10 * dt, dt_s=dt, objects={"bowl": {"pos": (-0.2, 0.0, 0.80)}},
        grippers={}, contacts=(("robot_a_left", "bowl"), ("robot_b_left", "bowl")),
    )
    res_bad = make_evaluator("bowl").evaluate_sequence(steps_bad)
    assert not res_bad.milestones["c_1"] and not res_bad.joint_success


def test_color_evaluator() -> None:
    dt = 0.04
    steps = [
        StepEvidence(0.0, dt, {
            "block_a_shared": {"pos": (-1.2, 0.0, 0.74), "side": "A"},
            "block_a_dist": {"pos": (-1.2, 0.2, 0.74), "side": "A"},
            "block_b_shared": {"pos": (1.2, 0.0, 0.74), "side": "B"},
        }, {"robot_a_left": 1.0, "robot_b_left": 1.0}, ()),
        StepEvidence(0.2, dt, {
            "block_a_shared": {"pos": (-1.2, 0.0, 0.80), "side": "A"},
            "block_a_dist": {"pos": (-1.2, 0.2, 0.74), "side": "A"},
            "block_b_shared": {"pos": (1.2, 0.0, 0.80), "side": "B"},
        }, {"robot_a_left": 0.0, "robot_b_left": 0.0}, (("robot_a_left", "block_a_shared"), ("robot_b_left", "block_b_shared"))),
        StepEvidence(0.5, dt, {
            "block_a_shared": {"pos": (-1.2, 0.0, 0.74), "side": "A", "in_regions": ("pad_A",)},
            "block_a_dist": {"pos": (-1.2, 0.2, 0.74), "side": "A"},
            "block_b_shared": {"pos": (1.2, 0.0, 0.74), "side": "B", "in_regions": ("pad_B",)},
        }, {"robot_a_left": 1.0, "robot_b_left": 1.0}, ()),
    ]
    res = make_evaluator("color").evaluate_sequence(steps)
    assert res.joint_success and res.S_collab == 1.0


def test_scan_evaluator() -> None:
    dt = 0.04
    steps = []
    for i in range(85):
        t = i * dt
        if i < 5:
            p_box, p_scn, ang = (-0.4, 0.0, 0.74), (0.4, 0.0, 0.74), 90.0
            c = (("tea_box", "table"), ("scanner", "table"))
        elif i < 30:
            p_box, p_scn, ang = (-0.04, 0.0, 0.84), (0.04, 0.0, 0.84), 5.0
            c = (("robot_a_left", "tea_box"), ("robot_b_right", "scanner"))
        elif i < 42:
            # B withdraws first (>30 mm away, leaves geometry) while A still holds box!
            p_box, p_scn, ang = (-0.04, 0.0, 0.84), (0.25, 0.0, 0.84), 55.0
            c = (("robot_a_left", "tea_box"), ("robot_b_right", "scanner"))
        elif i < 55:
            p_box, p_scn, ang = (-0.25, 0.0, 0.80), (0.4, 0.0, 0.74), 90.0
            c = (("robot_a_left", "tea_box"), ("scanner", "table"))
        else:
            p_box, p_scn, ang = (-0.4, 0.0, 0.74), (0.4, 0.0, 0.74), 90.0
            c = (("tea_box", "table"), ("scanner", "table"))
        steps.append(StepEvidence(t, dt, {"tea_box": {"pos": p_box}, "scanner": {"pos": p_scn, "angle_to_target_deg": ang}}, {}, c))
    res = make_evaluator("scan").evaluate_sequence(steps)
    assert res.joint_success and res.S_collab == 1.0


def test_door_evaluator() -> None:
    dt = 0.04
    steps = []
    for i in range(50):
        t = i * dt
        if i < 5:
            pot = {"pos": (0.1, -0.25, 0.75), "lin_vel": (0, 0, 0), "tilt_deg": 0.0, "in_regions": ()}
            door = {"joint_deg": 0.0}
            c = (("pot", "table"),)
        elif i < 25:
            pot = {"pos": (0.1, 0.05, 0.83), "lin_vel": (0, 0.2, 0), "tilt_deg": 0.0, "in_regions": ()}
            door = {"joint_deg": 148.0}
            c = (("panda_finger", "door"), ("aloha_left", "pot"), ("aloha_right", "pot"))
        else:
            pot = {"pos": (0.1, 0.25, 0.82), "lin_vel": (0, 0, 0.01), "tilt_deg": 0.0, "in_regions": ("cabinet_interior",)}
            door = {"joint_deg": 148.0}
            c = (("pot", "cabinet_shelf"),)
        steps.append(StepEvidence(t, dt, {"pot": pot, "door": door}, {}, c))
    res = make_evaluator("door").evaluate_sequence(steps)
    assert res.joint_success and res.S_role == 1.0 and res.S_collab == 1.0


def test_tray_food_balls_evaluators() -> None:
    dt = 0.04
    # Tray (7 milestones)
    t_steps = []
    for i in range(60):
        if i < 5:
            tray = {"pos": (0.0, 0.0, 0.75), "tilt_deg": 0.0, "in_regions": ()}
            c = (("tray", "table"),)
        elif i < 25:
            tray = {"pos": (0.05 * (i - 5) / 20.0, 0.0, 0.86), "tilt_deg": 1.0, "in_regions": ()}
            c = (("A.left", "tray"), ("A.right", "tray"), ("B.left", "tray"), ("B.right", "tray"))
        else:
            tray = {"pos": (0.05, 0.0, 0.75), "tilt_deg": 0.5, "in_regions": ("tray_goal_zone",)}
            c = (("tray", "table"),)
        objs = {"tray": tray, **{f"block_{k}": {"container": "tray"} for k in range(1, 5)}}
        t_steps.append(StepEvidence(i * dt, dt, objs, {}, c))
    r_tray = make_evaluator("tray").evaluate_sequence(t_steps)
    assert r_tray.joint_success and len(r_tray.milestones) == 7

    # Food (6 milestones)
    f_steps = [
        StepEvidence(0.0, dt, {"tray": {"pos": (0, 0, 0.75), "tilt_deg": 0.0}, "apple": {"container": "tray"}, "carrot": {"container": "tray"}, "cucumber": {"container": "tray"}}, {}, (("tray", "table"),)),
        StepEvidence(0.5, dt, {"tray": {"pos": (0, 0, 0.84), "tilt_deg": 1.0}, "apple": {"container": "tray"}, "carrot": {"container": "tray"}, "cucumber": {"container": "tray"}}, {}, (("A.left", "tray"), ("A.right", "tray"), ("B.left", "tray"), ("B.right", "tray"))),
        StepEvidence(1.0, dt, {"tray": {"pos": (0, 0, 0.84), "tilt_deg": 8.0}, "apple": {"container": "bag"}, "carrot": {"container": "bag"}, "cucumber": {"container": "bag"}}, {}, (("A.left", "tray"), ("A.right", "tray"), ("B.left", "tray"), ("B.right", "tray"))),
    ]
    r_food = make_evaluator("food").evaluate_sequence(f_steps)
    assert r_food.joint_success and len(r_food.milestones) == 6

    # Balls (6 milestones)
    b_steps = []
    for i in range(60):
        pouring = 5 <= i < 35
        objs = {
            "tipping_hopper": {"joint_deg": 22.0 if pouring else 0.0},
            "receiving_cup": {"pos": (0.4, 0.0, 0.75), "tilt_deg": 0.0, "in_regions": ("cup_target_zone",) if i >= 35 else ()},
            **{f"ball_{k}": {"container": "receiving_cup" if i >= 30 else "tipping_hopper"} for k in range(1, 4)},
        }
        c = (("panda_finger", "tipping_hopper"), ("ur5_finger", "receiving_cup")) if pouring else ()
        g = {"panda": 0.0 if pouring else 1.0, "ur5": 0.0 if pouring else 1.0}
        b_steps.append(StepEvidence(i * dt, dt, objs, g, c))
    r_balls = make_evaluator("balls").evaluate_sequence(b_steps)
    assert r_balls.joint_success and len(r_balls.milestones) == 6

    agg = aggregate_task_metrics([r_tray, r_food, r_balls])
    assert agg["SR"] == 100.0 and agg["trials"] == 3


def run_all() -> None:
    tests = [
        test_registry_counts_match_paper,
        test_wilson_ci_matches_paper_appendix,
        test_bowl_evaluator_pass_and_overlap_fail,
        test_color_evaluator,
        test_scan_evaluator,
        test_door_evaluator,
        test_tray_food_balls_evaluators,
    ]
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} MA-Bench evaluator tests passed.")


if __name__ == "__main__":
    run_all()
