"""Optimizer parameter grouping (Paper Section 4.3 & Appendix Table 6).

All weights are fine-tuned with constant LR = 1e-5 (AdamW, betas=(0.9, 0.95),
weight_decay=0.1), while global cross-attention gates `gamma_b` receive a 10x
learning rate (`1e-4`) with zero weight decay.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def classify_param(
    name: str,
    ndim: int,
    lr_shared: float = 1e-5,
    lr_global: float = 1e-5,
    lr_gate: float = 1e-4,
    weight_decay: float = 0.1,
) -> Tuple[str, float, float]:
    """Return `(family, lr, decay)` for a parameter `(name, ndim)`."""
    is_gate = name.endswith("global_gate")
    if is_gate:
        return ("gate", lr_gate, 0.0)
    is_global = "global_" in name
    lr = lr_global if is_global else lr_shared
    decay = 0.0 if (ndim <= 1 or name.endswith(".bias")) else weight_decay
    return ("global" if is_global else "shared", lr, decay)


def build_optimizer_groups_spec(
    named_ndims: Iterable[Tuple[str, int]],
    lr_shared: float = 1e-5,
    lr_global: float = 1e-5,
    lr_gate: float = 1e-4,
    weight_decay: float = 0.1,
) -> List[Dict[str, Any]]:
    """Pure-Python group builder for CPU verification without PyTorch."""
    buckets: Dict[Tuple[str, float, float], Dict[str, Any]] = {}
    for name, ndim in named_ndims:
        key = classify_param(name, ndim, lr_shared, lr_global, lr_gate, weight_decay)
        family, lr, decay = key
        bucket = buckets.setdefault(
            key,
            {"param_names": [], "lr": lr, "weight_decay": decay, "group_name": family},
        )
        bucket["param_names"].append(name)
    return list(buckets.values())


def build_optimizer_groups(
    model,
    weight_decay: float = 0.1,
    lr_shared: float = 1e-5,
    lr_global: float = 1e-5,
    lr_gate: float = 1e-4,
) -> List[Dict[str, Any]]:
    """Build PyTorch AdamW parameter groups matching Table 6."""
    buckets: Dict[Tuple[str, float, float], Dict[str, Any]] = {}
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        assert id(parameter) not in seen, name
        seen.add(id(parameter))
        key = classify_param(name, parameter.ndim, lr_shared, lr_global, lr_gate, weight_decay)
        family, lr, decay = key
        bucket = buckets.setdefault(
            key,
            {
                "params": [],
                "param_names": [],
                "lr": lr,
                "weight_decay": decay,
                "group_name": family,
            },
        )
        bucket["params"].append(parameter)
        bucket["param_names"].append(name)
    assert seen == {id(p) for p in model.parameters() if p.requires_grad}
    return list(buckets.values())
