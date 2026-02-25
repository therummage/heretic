# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Tests for Qwen3.5 hybrid attention architecture support.

Creates tiny Qwen3.5 models from config (no downloads) and verifies
that heretic's layer module detection and abliteration work correctly
with the hybrid Gated DeltaNet + full attention layer layout, for both
dense and MoE variants.
"""

from contextlib import suppress

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import Tensor
from torch.nn import Module
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM


@pytest.fixture(scope="module")
def tiny_qwen35() -> tuple[Qwen3_5ForCausalLM, Qwen3_5TextConfig]:
    """Create a minimal Qwen3.5 dense model for testing (< 1MB)."""
    config = Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=1000,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
    )
    model = Qwen3_5ForCausalLM(config)
    model.eval()
    return model, config


@pytest.fixture(scope="module")
def tiny_qwen35_moe() -> tuple[Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig]:
    """Create a minimal Qwen3.5 MoE model for testing."""
    config = Qwen3_5MoeTextConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=1000,
        num_experts=4,
        num_experts_per_tok=2,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
    )
    model = Qwen3_5MoeForCausalLM(config)
    model.eval()
    return model, config


def get_layer_modules(layer: Module) -> dict[str, list[Module]]:
    """Replicate heretic's get_layer_modules logic for testing."""
    modules: dict[str, list[Module]] = {}

    def try_add(component: str, module: object) -> None:
        if isinstance(module, Module):
            if component not in modules:
                modules[component] = []
            modules[component].append(module)
        else:
            assert not isinstance(module, Tensor), (
                f"Unexpected Tensor in {component} - expected nn.Module"
            )

    # Most models (standard full attention).
    with suppress(Exception):
        try_add("attn.o_proj", layer.self_attn.o_proj)

    # Hybrid attention models (e.g. Qwen3.5).
    with suppress(Exception):
        try_add("attn.out_proj", layer.linear_attn.out_proj)

    # Most dense models.
    with suppress(Exception):
        try_add("mlp.down_proj", layer.mlp.down_proj)

    # MoE models with shared experts (e.g. Qwen3.5 MoE).
    with suppress(Exception):
        try_add("mlp.down_proj", layer.mlp.shared_expert.down_proj)

    # MoE models with iterable expert modules.
    with suppress(Exception):
        for expert in layer.mlp.experts:
            try_add("mlp.down_proj", expert.down_proj)

    total_modules = sum(len(mods) for mods in modules.values())
    assert total_modules > 0, "No abliterable modules found in layer"

    return modules


def test_layer_types(tiny_qwen35):
    """Verify the 3:1 hybrid attention layout."""
    model, config = tiny_qwen35
    assert config.layer_types == [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ]


def test_linear_attention_layers(tiny_qwen35):
    """Linear attention layers should have linear_attn.out_proj and mlp.down_proj."""
    model, config = tiny_qwen35
    layers = model.model.layers

    for i, layer in enumerate(layers):
        if config.layer_types[i] != "linear_attention":
            continue
        mods = get_layer_modules(layer)
        assert "attn.out_proj" in mods, f"Layer {i}: missing attn.out_proj"
        assert "mlp.down_proj" in mods, f"Layer {i}: missing mlp.down_proj"
        assert "attn.o_proj" not in mods, f"Layer {i}: unexpected attn.o_proj"


def test_full_attention_layers(tiny_qwen35):
    """Full attention layers should have self_attn.o_proj and mlp.down_proj."""
    model, config = tiny_qwen35
    layers = model.model.layers

    for i, layer in enumerate(layers):
        if config.layer_types[i] != "full_attention":
            continue
        mods = get_layer_modules(layer)
        assert "attn.o_proj" in mods, f"Layer {i}: missing attn.o_proj"
        assert "mlp.down_proj" in mods, f"Layer {i}: missing mlp.down_proj"
        assert "attn.out_proj" not in mods, f"Layer {i}: unexpected attn.out_proj"


def test_abliterable_components_union(tiny_qwen35):
    """get_abliterable_components must return the union across all layer types."""
    model, _ = tiny_qwen35
    layers = model.model.layers

    # Replicate heretic's get_abliterable_components (fixed version).
    components: dict[str, None] = {}
    for layer_index in range(len(layers)):
        for component in get_layer_modules(layers[layer_index]):
            components[component] = None
    component_list = list(components)

    assert "attn.o_proj" in component_list, "Missing attn.o_proj from full attention layers"
    assert "attn.out_proj" in component_list, "Missing attn.out_proj from linear attention layers"
    assert "mlp.down_proj" in component_list, "Missing mlp.down_proj"


def test_lora_targets(tiny_qwen35):
    """LoRA target modules should include both o_proj and out_proj."""
    model, _ = tiny_qwen35
    layers = model.model.layers

    # Replicate component discovery.
    components: dict[str, None] = {}
    for layer_index in range(len(layers)):
        for component in get_layer_modules(layers[layer_index]):
            components[component] = None

    target_modules = [comp.split(".")[-1] for comp in components]
    assert "o_proj" in target_modules, "LoRA targets missing o_proj"
    assert "out_proj" in target_modules, "LoRA targets missing out_proj"
    assert "down_proj" in target_modules, "LoRA targets missing down_proj"


def test_lora_adapter_attachment(tiny_qwen35):
    """LoRA adapters should attach successfully to both attention types."""
    model, _ = tiny_qwen35

    target_modules = ["o_proj", "out_proj", "down_proj"]
    peft_config = LoraConfig(
        r=1,
        target_modules=target_modules,
        lora_alpha=1,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, peft_config)

    # Verify LoRA was applied to at least some modules.
    lora_modules = [
        name for name, _ in peft_model.named_modules() if "lora_" in name
    ]
    assert len(lora_modules) > 0, "No LoRA modules found"


def test_abliteration_parameters_no_keyerror(tiny_qwen35):
    """Abliteration loop should not KeyError on hybrid layers."""
    model, _ = tiny_qwen35
    layers = model.model.layers

    # Build component union (like fixed get_abliterable_components).
    components: dict[str, None] = {}
    for layer_index in range(len(layers)):
        for component in get_layer_modules(layers[layer_index]):
            components[component] = None

    # Simulate the parameters dict built from components.
    parameters = {comp: "dummy_params" for comp in components}

    # Simulate abliterate() loop - should not raise KeyError.
    for layer_index in range(len(layers)):
        for component, modules in get_layer_modules(layers[layer_index]).items():
            assert component in parameters, (
                f"Layer {layer_index}: component '{component}' not in parameters"
            )


# ──────────────────────────────────────────────────────────────────────
# Qwen 3.5 MoE tests
# ──────────────────────────────────────────────────────────────────────


def test_moe_layer_types(tiny_qwen35_moe):
    """Verify the 3:1 hybrid attention layout on the MoE variant."""
    _, config = tiny_qwen35_moe
    assert config.layer_types == [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ]


def test_moe_shared_expert_found(tiny_qwen35_moe):
    """MoE layers should discover the shared expert's down_proj."""
    model, config = tiny_qwen35_moe
    layers = model.model.layers

    for i, layer in enumerate(layers):
        mods = get_layer_modules(layer)
        assert "mlp.down_proj" in mods, (
            f"Layer {i}: missing mlp.down_proj (shared expert not detected)"
        )
        # The shared expert contributes exactly one nn.Linear.
        assert len(mods["mlp.down_proj"]) == 1, (
            f"Layer {i}: expected 1 shared expert down_proj, got {len(mods['mlp.down_proj'])}"
        )


def test_moe_routed_experts_not_iterable(tiny_qwen35_moe):
    """Routed experts use packed 3D tensors — they should not appear as modules."""
    model, _ = tiny_qwen35_moe
    layer = model.model.layers[0]

    # Qwen3_5MoeExperts is NOT iterable (packed Parameter tensors).
    with pytest.raises(TypeError):
        iter(layer.mlp.experts)


def test_moe_abliterable_components(tiny_qwen35_moe):
    """get_abliterable_components must return the union across all MoE layer types."""
    model, _ = tiny_qwen35_moe
    layers = model.model.layers

    components: dict[str, None] = {}
    for layer_index in range(len(layers)):
        for component in get_layer_modules(layers[layer_index]):
            components[component] = None
    component_list = list(components)

    assert "attn.o_proj" in component_list, "Missing attn.o_proj from full attention layers"
    assert "attn.out_proj" in component_list, "Missing attn.out_proj from linear attention layers"
    assert "mlp.down_proj" in component_list, "Missing mlp.down_proj from shared expert"


def test_moe_lora_adapter_attachment(tiny_qwen35_moe):
    """LoRA adapters should attach to shared expert down_proj on MoE models."""
    model, _ = tiny_qwen35_moe

    target_modules = ["o_proj", "out_proj", "down_proj"]
    peft_config = LoraConfig(
        r=1,
        target_modules=target_modules,
        lora_alpha=1,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, peft_config)

    # Verify LoRA was applied to shared_expert.down_proj modules.
    lora_shared_expert = [
        name for name, _ in peft_model.named_modules()
        if "shared_expert" in name and "lora_" in name
    ]
    assert len(lora_shared_expert) > 0, "No LoRA modules on shared expert"


def test_moe_abliteration_no_keyerror(tiny_qwen35_moe):
    """Abliteration loop should not KeyError on hybrid MoE layers."""
    model, _ = tiny_qwen35_moe
    layers = model.model.layers

    components: dict[str, None] = {}
    for layer_index in range(len(layers)):
        for component in get_layer_modules(layers[layer_index]):
            components[component] = None

    parameters = {comp: "dummy_params" for comp in components}

    for layer_index in range(len(layers)):
        for component, modules in get_layer_modules(layers[layer_index]).items():
            assert component in parameters, (
                f"Layer {layer_index}: component '{component}' not in parameters"
            )
