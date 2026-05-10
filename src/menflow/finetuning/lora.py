"""LoRA adapter integration via :mod:`peft`.

Uses ``peft.inject_adapter_in_model`` to patch attention projection linears
(``to_q``, ``to_k``, ``to_v``, ``out_proj``) in-place. The host model object is
unchanged in identity, so the volume-conditional monkey-patch on
``unet._get_time_and_class_embedding`` survives.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
arXiv:2106.09685.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer
from torch import nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    """Routine-side LoRA hyperparameters; converted to a `peft.LoraConfig` at apply time."""

    rank: int = 8
    alpha: float = 8.0
    dropout: float = 0.0
    # peft uses suffix matching on `target_modules` — exactly the names produced
    # by MAISI's attention blocks: `…attn.to_q`, `…attn.to_k`, `…attn.to_v`,
    # `…attn.out_proj`.
    target_modules: tuple[str, ...] = ("to_q", "to_k", "to_v", "out_proj")
    bias: str = "none"  # peft option; "none" = only A,B trainable


def _to_peft_config(cfg: LoRAConfig) -> LoraConfig:
    return LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        bias=cfg.bias,
        target_modules=list(cfg.target_modules),
        # `init_lora_weights=True` is peft's default: A is kaiming-normal, B is
        # zero — required for the identity-at-init invariant we assert in the
        # engine.
    )


def apply_lora(model: nn.Module, *, cfg: LoRAConfig) -> dict[str, LoraLayer]:
    """Inject LoRA adapters into `model` in place.

    Returns a mapping of dotted module path → adapter for downstream
    introspection (size, layer count). The base model's parameter
    ``requires_grad`` flags are reset so that only the LoRA A/B weights are
    trainable; any modules outside `model` (e.g. the volume conditioner) must
    have their `requires_grad` re-enabled by the caller after this call.
    """
    inject_adapter_in_model(_to_peft_config(cfg), model)
    adapters: dict[str, LoraLayer] = {}
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer):
            adapters[name] = m
    if not adapters:
        raise RuntimeError(
            f"peft.inject_adapter_in_model targeted no modules; check that "
            f"target_modules={cfg.target_modules!r} suffix-matches names in this model"
        )
    # Freeze everything that isn't a LoRA A/B parameter.
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
    logger.info(
        "peft injected %d LoRA layers (rank=%d, alpha=%.1f) targeting %s",
        len(adapters),
        cfg.rank,
        cfg.alpha,
        list(cfg.target_modules),
    )
    return adapters


def collect_lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only the LoRA-trainable parameters as a state-dict-style mapping.

    Uses `state_dict()` keys directly so they match `named_parameters()` and
    can be loaded back with :func:`load_lora_state` without renaming.
    """
    sd = model.state_dict()
    return {k: v.detach().cpu() for k, v in sd.items() if "lora_" in k}


def load_lora_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Copy LoRA weights from `state` into the matching parameters of `model`."""
    own = dict(model.named_parameters())
    for k, v in state.items():
        if k not in own:
            continue
        with torch.no_grad():
            own[k].copy_(v.to(own[k].device))


def lora_param_iter(model: nn.Module) -> Iterable[nn.Parameter]:
    """Yield only the trainable LoRA parameters of `model`."""
    for n, p in model.named_parameters():
        if p.requires_grad and ("lora_A" in n or "lora_B" in n):
            yield p
