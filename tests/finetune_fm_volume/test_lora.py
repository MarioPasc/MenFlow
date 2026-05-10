"""Tests for the peft-based LoRA integration in
:mod:`routines.finetune_fm_volume.engine.lora`.

The custom ``LoRALinear`` is gone — peft's ``inject_adapter_in_model`` is the
sole way LoRA enters the model. These tests verify (a) that injection actually
hits the requested target submodules, (b) that non-LoRA parameters are frozen,
and (c) that the state-dict round-trip is loss-less.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

peft = pytest.importorskip("peft")
from peft.tuners.lora import LoraLayer  # noqa: E402

from menflow.finetuning.lora import (  # noqa: E402
    LoRAConfig,
    apply_lora,
    collect_lora_state,
    load_lora_state,
)


class _AttnHost(nn.Module):
    """Tiny host with attention-shaped submodule names."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(8, 8)
        self.attn.to_k = nn.Linear(8, 8)
        self.attn.to_v = nn.Linear(8, 8)
        self.attn.out_proj = nn.Linear(8, 8)
        self.ffn = nn.Linear(8, 8)  # outside the target list; must stay frozen
        self.norm = nn.LayerNorm(8)


def test_apply_lora_targets_only_attention_linears() -> None:
    model = _AttnHost()
    cfg = LoRAConfig(rank=4, alpha=4.0)
    adapters = apply_lora(model, cfg=cfg)
    matched = {n for n in adapters}
    expected_suffixes = {"attn.to_q", "attn.to_k", "attn.to_v", "attn.out_proj"}
    assert all(any(n.endswith(s) for n in matched) for s in expected_suffixes), matched


def test_lora_freezes_non_target_params() -> None:
    model = _AttnHost()
    apply_lora(model, cfg=LoRAConfig(rank=4, alpha=4.0))
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            assert p.requires_grad, n
        else:
            assert not p.requires_grad, n


def test_lora_identity_at_init() -> None:
    """peft init: A is kaiming, B is zero ⇒ wrapped forward ≡ base forward."""
    torch.manual_seed(0)
    model = _AttnHost()
    base_q = model.attn.to_q
    x = torch.randn(2, 8)
    ref = base_q(x).clone()
    apply_lora(model, cfg=LoRAConfig(rank=4, alpha=4.0))
    out = model.attn.to_q(x)
    assert torch.allclose(out, ref, atol=1e-6), (
        "B=0 init must keep peft-LoRA identity to base linear"
    )


def test_lora_state_round_trip() -> None:
    torch.manual_seed(1)
    model = _AttnHost()
    apply_lora(model, cfg=LoRAConfig(rank=2, alpha=2.0))
    # Mutate so the round-trip is non-trivial.
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.data.normal_()
    state = collect_lora_state(model)
    assert len(state) > 0, "peft state dict must include LoRA weights"

    model2 = _AttnHost()
    apply_lora(model2, cfg=LoRAConfig(rank=2, alpha=2.0))
    load_lora_state(model2, state)
    own = dict(model.named_parameters())
    own2 = dict(model2.named_parameters())
    for k in own:
        if "lora_" in k:
            assert torch.allclose(own[k], own2[k]), k


def test_lora_layer_count_scales_with_targets() -> None:
    """A model with 4 target linears must yield 4 LoraLayer instances."""
    model = _AttnHost()
    adapters = apply_lora(model, cfg=LoRAConfig(rank=4, alpha=4.0))
    assert len(adapters) == 4
    for m in adapters.values():
        assert isinstance(m, LoraLayer)
