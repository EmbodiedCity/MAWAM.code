"""Canonical registry of MA-Bench (7 simulation tasks) and 3 real-robot tasks.

All taxonomy labels, robot teams, milestone definitions, demonstration counts,
episode durations, and Hugging Face checkpoint paths strictly match Paper
Section 3.3, Section 5, Appendix Table 5, and `EmbodiedCity/MAWAM`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class RobotSlotSpec:
    agent_id: int
    role_key: str        # "A" or "B"
    embodiment: str      # "aloha" | "panda" | "ur5" | "franka_fr3" | "cobot_magic"
    arms: int            # 1 or 2
    action_dim: int      # 8, 14, or 16
    missing_right_wrist: bool
    cameras: Tuple[str, ...] = ("head", "left_wrist", "right_wrist")


@dataclass(frozen=True)
class TaskInfo:
    slug: str
    alias: str
    title: str
    why: str                     # information | spatial | capability
    when: str                    # asynchronous | partial | synchronous
    who: str                     # homogeneous | heterogeneous
    robots: Tuple[RobotSlotSpec, ...]
    nominal_duration_s: float
    demo_pairs: int
    eval_trials: int
    control_hz: float
    physics_hz: float
    role_milestones_A: Tuple[str, ...]
    role_milestones_B: Tuple[str, ...]
    collab_milestones: Tuple[str, ...]
    global_prompt: str
    local_prompts: Tuple[str, ...]
    hf_checkpoint_subdir: str
    taskspec_filename: str = ""
    action_clip: float = 5.0
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def num_agents(self) -> int:
        return len(self.robots)

    @property
    def total_milestones(self) -> int:
        return len(self.role_milestones_A) + len(self.role_milestones_B) + len(self.collab_milestones)


_ALOHA_A = RobotSlotSpec(0, "A", "aloha", arms=2, action_dim=16, missing_right_wrist=False)
_ALOHA_B = RobotSlotSpec(1, "B", "aloha", arms=2, action_dim=16, missing_right_wrist=False)
_PANDA_A = RobotSlotSpec(0, "A", "panda", arms=1, action_dim=8, missing_right_wrist=True)
_UR5_B = RobotSlotSpec(1, "B", "ur5", arms=1, action_dim=8, missing_right_wrist=True)

_FRANKA_REAL_A = RobotSlotSpec(0, "A", "franka_fr3", arms=1, action_dim=8, missing_right_wrist=False)
_COBOT_REAL_B = RobotSlotSpec(1, "B", "cobot_magic", arms=2, action_dim=14, missing_right_wrist=False)


MABENCH_TASKS: Dict[str, TaskInfo] = {
    "joint_bowl_placing": TaskInfo(
        slug="joint_bowl_placing",
        alias="bowl",
        title="Joint bowl placing",
        why="spatial",
        when="asynchronous",
        who="homogeneous",
        robots=(_ALOHA_A, _ALOHA_B),
        nominal_duration_s=90.0,
        demo_pairs=100,
        eval_trials=100,
        control_hz=25.0,
        physics_hz=250.0,
        role_milestones_A=("m_A1", "m_A2"),
        role_milestones_B=("m_B1", "m_B2"),
        collab_milestones=("c_1", "c_2"),
        global_prompt="Two robots relay the bowl from the start zone through the shared zone to the goal pad.",
        local_prompts=(
            "Grasp the bowl on your side, place it stably in the shared exchange zone, and retreat.",
            "Wait until the bowl rests in the shared zone, grasp it, place it on the goal pad, and retreat.",
        ),
        hf_checkpoint_subdir="ckpt/MAWAM-Joint-Bowl-Placing",
        taskspec_filename="joint_bowl_placing.yaml",
    ),
    "find_common_color": TaskInfo(
        slug="find_common_color",
        alias="color",
        title="Find common color",
        why="information",
        when="asynchronous",
        who="homogeneous",
        robots=(_ALOHA_A, _ALOHA_B),
        nominal_duration_s=13.0,
        demo_pairs=100,
        eval_trials=100,
        control_hz=25.0,
        physics_hz=250.0,
        role_milestones_A=("m_A1", "m_A2"),
        role_milestones_B=("m_B1", "m_B2"),
        collab_milestones=("c_1", "c_2"),
        global_prompt="Two robots 2.7 m apart identify the single block color shared across both sides and place it on their respective pads.",
        local_prompts=(
            "You are robot A. Pick the block whose color also appears on robot B's side and place it on your pad.",
            "You are robot B. Pick the block whose color also appears on robot A's side and place it on your pad.",
        ),
        hf_checkpoint_subdir="ckpt/MAWAM-Find-Common-Color",
        taskspec_filename="find_common_color.yaml",
    ),
    "hold_and_scan": TaskInfo(
        slug="hold_and_scan",
        alias="scan",
        title="Hold and scan",
        why="spatial",
        when="partial",
        who="homogeneous",
        robots=(_ALOHA_A, _ALOHA_B),
        nominal_duration_s=56.0,
        demo_pairs=100,
        eval_trials=100,
        control_hz=25.0,
        physics_hz=250.0,
        role_milestones_A=("m_A1", "m_A2"),
        role_milestones_B=("m_B1", "m_B2"),
        collab_milestones=("c_1", "c_2"),
        global_prompt="One robot presents the product box in the shared zone while the other scans its barcode, then both return their objects.",
        local_prompts=(
            "Grasp the box, lift and hold it steady facing the shared zone until scanning finishes, then return it.",
            "Grasp the scanner, align it with the presented box for at least 0.6 s, withdraw first, and return the scanner.",
        ),
        hf_checkpoint_subdir="ckpt/MAWAM-Hold-and-Scan",
        taskspec_filename="hold_and_scan.yaml",
    ),
    "open_door_place_pot": TaskInfo(
        slug="open_door_place_pot",
        alias="door",
        title="Open door, place pot",
        why="capability",
        when="partial",
        who="heterogeneous",
        robots=(_PANDA_A, _ALOHA_B),
        nominal_duration_s=62.08,
        demo_pairs=100,
        eval_trials=100,
        control_hz=25.0,
        physics_hz=250.0,
        role_milestones_A=("m_A1", "m_A2"),
        role_milestones_B=("m_B1", "m_B2"),
        collab_milestones=("c_1", "c_2"),
        global_prompt="Two robots put the pot into the cabinet together: the single arm opens the door and the dual arm places the pot.",
        local_prompts=(
            "Open the cabinet door and hold it open.",
            "Lift the pot with both hands and place it inside the cabinet.",
        ),
        hf_checkpoint_subdir="ckpt/MAWAM-Open-Door-Place-Pot",
        taskspec_filename="open_door_place_pot.yaml",
    ),
    "lift_tray_keep_blocks": TaskInfo(
        slug="lift_tray_keep_blocks",
        alias="tray",
        title="Lift tray, keep blocks",
        why="capability",
        when="synchronous",
        who="homogeneous",
        robots=(_ALOHA_A, _ALOHA_B),
        nominal_duration_s=40.0,
        demo_pairs=100,
        eval_trials=100,
        control_hz=25.0,
        physics_hz=250.0,
        role_milestones_A=("m_A1", "m_A2"),
        role_milestones_B=("m_B1", "m_B2"),
        collab_milestones=("c_1", "c_2", "c_3"),
        global_prompt="Two dual-arm robots grasp opposite sides of the tray with four hands, transport it without toppling the block stack, and set it down.",
        local_prompts=(
            "Grasp the two tray handles on your side, lift and transport the tray levelly with robot B, and release after set-down.",
            "Grasp the two tray handles on your side, lift and transport the tray levelly with robot A, and release after set-down.",
        ),
        hf_checkpoint_subdir="ckpt/MAWAM-Lift-Tray-Keep-Blocks",
        taskspec_filename="lift_tray_keep_blocks.yaml",
    ),
    "lift_tray_pour_food": TaskInfo(
        slug="lift_tray_pour_food",
        alias="food",
        title="Lift tray, pour food",
        why="capability",
        when="synchronous",
        who="homogeneous",
        robots=(_ALOHA_A, _ALOHA_B),
        nominal_duration_s=39.0,
        demo_pairs=100,
        eval_trials=100,
        control_hz=25.0,
        physics_hz=250.0,
        role_milestones_A=("m_A1", "m_A2"),
        role_milestones_B=("m_B1", "m_B2"),
        collab_milestones=("c_1", "c_2"),
        global_prompt="The two robots put everything into the garbage bag together.",
        local_prompts=(
            "You are robot A. Grasp the side of the tray near you with both arms; together with robot B, lift the tray, move it to the table edge, tilt it to pour all three items into the open bag, and keep holding the tray until the food has settled.",
            "You are robot B. Grasp the side of the tray near you with both arms; together with robot A, lift the tray, move it to the table edge, tilt it to pour all three items into the open bag, and keep holding the tray until the food has settled.",
        ),
        hf_checkpoint_subdir="ckpt/MAWAM-Lift-Tray-Pour-Food",
        taskspec_filename="lift_tray_pour_food.yaml",
    ),
    "pour_and_catch": TaskInfo(
        slug="pour_and_catch",
        alias="balls",
        title="Pour and catch",
        why="capability",
        when="synchronous",
        who="heterogeneous",
        robots=(_PANDA_A, _UR5_B),
        nominal_duration_s=38.0,
        demo_pairs=100,
        eval_trials=100,
        control_hz=25.0,
        physics_hz=250.0,
        role_milestones_A=("m_A1", "m_A2"),
        role_milestones_B=("m_B1", "m_B2"),
        collab_milestones=("c_1", "c_2"),
        global_prompt="Panda tilts the hopper to pour three balls while UR5 tracks the moving outlet with a cup and sets the cup on the target.",
        local_prompts=(
            "Grasp the hopper handle, tilt the hopper to pour out the balls, and return the hopper upright.",
            "Grasp the receiving cup, raise and track under the moving hopper outlet to catch all balls, and place the cup on the target.",
        ),
        hf_checkpoint_subdir="ckpt/MAWAM-Pour-and-Catch",
        taskspec_filename="pour_and_catch.yaml",
    ),
}


REAL_ROBOT_TASKS: Dict[str, TaskInfo] = {
    "fruit_basket": TaskInfo(
        slug="fruit_basket",
        alias="fruit",
        title="Fruit basket",
        why="spatial",
        when="partial",
        who="heterogeneous",
        robots=(_FRANKA_REAL_A, _COBOT_REAL_B),
        nominal_duration_s=30.0,
        demo_pairs=80,
        eval_trials=20,
        control_hz=30.0,
        physics_hz=30.0,
        role_milestones_A=("m_A1", "m_A2"),
        role_milestones_B=("m_B1", "m_B2"),
        collab_milestones=("c_1", "c_2"),
        global_prompt="Cobot holds and tilts the basket while Franka places the green apple, peach, and banana bunch inside.",
        local_prompts=(
            "Pick each fruit on the table and place it into the basket held by the dual-arm robot.",
            "Grasp both sides of the basket, lift and hold it steady until all fruits are placed, then set it down.",
        ),
        hf_checkpoint_subdir="",
    ),
    "toy_handoff_sort": TaskInfo(
        slug="toy_handoff_sort",
        alias="toy",
        title="Toy handoff & sort",
        why="spatial",
        when="synchronous",
        who="heterogeneous",
        robots=(_FRANKA_REAL_A, _COBOT_REAL_B),
        nominal_duration_s=30.0,
        demo_pairs=80,
        eval_trials=20,
        control_hz=30.0,
        physics_hz=30.0,
        role_milestones_A=("m_A1", "m_A2"),
        role_milestones_B=("m_B1", "m_B2"),
        collab_milestones=("c_1", "c_2"),
        global_prompt="Franka lifts plush toys for mid-air handoff to Cobot, which sorts each category into its target bin.",
        local_prompts=(
            "Pick each plush toy and hand it over mid-air to the dual-arm robot.",
            "Receive each toy mid-air from the single-arm robot and sort it into the matching bin.",
        ),
        hf_checkpoint_subdir="",
    ),
    "funnel_cube_pour": TaskInfo(
        slug="funnel_cube_pour",
        alias="funnel",
        title="Funnel cube pour",
        why="capability",
        when="synchronous",
        who="heterogeneous",
        robots=(_FRANKA_REAL_A, _COBOT_REAL_B),
        nominal_duration_s=30.0,
        demo_pairs=80,
        eval_trials=20,
        control_hz=30.0,
        physics_hz=30.0,
        role_milestones_A=("m_A1", "m_A2"),
        role_milestones_B=("m_B1", "m_B2"),
        collab_milestones=("c_1", "c_2"),
        global_prompt="Cobot coaxially stabilizes the handled funnel over the cup while Franka pours wooden cubes from the bowl.",
        local_prompts=(
            "Lift the bowl of wooden cubes and pour them smoothly into the stabilized funnel.",
            "Hold the handled funnel coaxially over the receiving cup until pouring finishes.",
        ),
        hf_checkpoint_subdir="",
    ),
}

_ALIAS_MAP: Dict[str, str] = {info.alias: slug for slug, info in MABENCH_TASKS.items()}
_ALIAS_MAP.update({info.alias: slug for slug, info in REAL_ROBOT_TASKS.items()})


def get_task_info(name: str) -> TaskInfo:
    key = name.strip().lower()
    if key in MABENCH_TASKS:
        return MABENCH_TASKS[key]
    if key in REAL_ROBOT_TASKS:
        return REAL_ROBOT_TASKS[key]
    if key in _ALIAS_MAP:
        resolved = _ALIAS_MAP[key]
        return MABENCH_TASKS.get(resolved) or REAL_ROBOT_TASKS[resolved]
    raise KeyError(
        f"Unknown task {name!r}; expected one of {sorted(MABENCH_TASKS)} or aliases {sorted(_ALIAS_MAP)}"
    )
