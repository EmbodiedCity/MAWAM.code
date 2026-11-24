"""Deterministic, LLM-free MA-Bench evaluators for all seven tasks (43 milestones)."""

from ma_bench.evaluators.base import BaseTaskEvaluator, StepEvidence
from ma_bench.evaluators.balls import PourAndCatchEvaluator
from ma_bench.evaluators.bowl import JointBowlPlacingEvaluator
from ma_bench.evaluators.color import FindCommonColorEvaluator
from ma_bench.evaluators.door import OpenDoorPlacePotEvaluator
from ma_bench.evaluators.food import LiftTrayPourFoodEvaluator
from ma_bench.evaluators.scan import HoldAndScanEvaluator
from ma_bench.evaluators.tray import LiftTrayKeepBlocksEvaluator

EVALUATOR_REGISTRY = {
    "joint_bowl_placing": JointBowlPlacingEvaluator,
    "find_common_color": FindCommonColorEvaluator,
    "hold_and_scan": HoldAndScanEvaluator,
    "open_door_place_pot": OpenDoorPlacePotEvaluator,
    "lift_tray_keep_blocks": LiftTrayKeepBlocksEvaluator,
    "lift_tray_pour_food": LiftTrayPourFoodEvaluator,
    "pour_and_catch": PourAndCatchEvaluator,
}


def make_evaluator(task_slug: str, **kwargs) -> BaseTaskEvaluator:
    from ma_bench.registry import get_task_info

    slug = get_task_info(task_slug).slug
    return EVALUATOR_REGISTRY[slug](**kwargs)


__all__ = [
    "EVALUATOR_REGISTRY",
    "BaseTaskEvaluator",
    "StepEvidence",
    "JointBowlPlacingEvaluator",
    "FindCommonColorEvaluator",
    "HoldAndScanEvaluator",
    "OpenDoorPlacePotEvaluator",
    "LiftTrayKeepBlocksEvaluator",
    "LiftTrayPourFoodEvaluator",
    "PourAndCatchEvaluator",
    "make_evaluator",
]
