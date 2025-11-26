"""
Expert routing intervention V3 - Modify router bias directly.

Instead of modifying router logits after computation, we modify the router's bias parameter
BEFORE the forward pass. This ensures routing structures remain consistent.
"""

import torch
from typing import Dict, Tuple

class ExpertInterventionConfig:
    """Configuration for expert routing interventions."""

    def __init__(self):
        self.interventions: Dict[Tuple[int, int], float] = {}

    def force_expert(self, layer: int, expert_id: int, strength: float = 10.0):
        """Force an expert by boosting its bias."""
        self.interventions[(layer, expert_id)] = strength
        return self

    def suppress_expert(self, layer: int, expert_id: int, strength: float = -10.0):
        """Suppress an expert by reducing its bias."""
        self.interventions[(layer, expert_id)] = strength
        return self

    def soft_bias_expert(self, layer: int, expert_id: int, strength: float = 2.0):
        """Soft bias towards an expert."""
        self.interventions[(layer, expert_id)] = strength
        return self

    def get_interventions_for_layer(self, layer_idx: int) -> Dict[int, float]:
        """Get interventions for a specific layer."""
        return {
            expert_id: strength
            for (layer, expert_id), strength in self.interventions.items()
            if layer == layer_idx
        }


def apply_expert_interventions(model_base, config: ExpertInterventionConfig):
    """
    Apply interventions by modifying router biases.

    Args:
        model_base: The model wrapper
        config: Intervention configuration

    Returns:
        Dict mapping layer_idx -> original bias for restoration
    """
    original_biases = {}

    layers_to_intervene = set(layer for layer, _ in config.interventions.keys())

    for layer_idx in layers_to_intervene:
        interventions = config.get_interventions_for_layer(layer_idx)
        if not interventions:
            continue

        router = model_base.model.model.layers[layer_idx].mlp.router

        # Save original bias
        original_biases[layer_idx] = router.bias.data.clone()

        # Modify bias
        modified_bias = router.bias.data.clone()
        for expert_id, strength in interventions.items():
            modified_bias[expert_id] += strength

        router.bias.data = modified_bias

    return original_biases


def remove_expert_interventions(model_base, original_biases):
    """
    Remove interventions by restoring original router biases.

    Args:
        model_base: The model wrapper
        original_biases: Dict from apply_expert_interventions
    """
    for layer_idx, original_bias in original_biases.items():
        router = model_base.model.model.layers[layer_idx].mlp.router
        router.bias.data = original_bias


def print_intervention_summary(config: ExpertInterventionConfig):
    """Print intervention summary."""
    print("Expert Routing Intervention Summary (Bias Modification)")
    print("=" * 60)

    force_experts = [(l, e, s) for (l, e), s in config.interventions.items() if s > 5]
    suppress_experts = [(l, e, s) for (l, e), s in config.interventions.items() if s < -5]
    soft_experts = [(l, e, s) for (l, e), s in config.interventions.items() if -5 <= s <= 5]

    if force_experts:
        print("\nForced Experts (bias +10):")
        for layer, expert, strength in force_experts:
            print(f"  Layer {layer}, Expert {expert}: +{strength}")

    if suppress_experts:
        print("\nSuppressed Experts (bias -10):")
        for layer, expert, strength in suppress_experts:
            print(f"  Layer {layer}, Expert {expert}: {strength}")

    if soft_experts:
        print("\nSoft Bias Experts:")
        for layer, expert, strength in soft_experts:
            print(f"  Layer {layer}, Expert {expert}: {strength:+.1f}")

    if not config.interventions:
        print("No interventions configured")

    print("=" * 60)


# Pre-configured experiments
def get_harmful_expert_suppression_config():
    config = ExpertInterventionConfig()
    config.suppress_expert(layer=10, expert_id=5, strength=-10.0)
    config.suppress_expert(layer=13, expert_id=1, strength=-10.0)
    return config


def get_harmful_expert_forcing_config():
    config = ExpertInterventionConfig()
    config.force_expert(layer=10, expert_id=5, strength=10.0)
    config.force_expert(layer=13, expert_id=1, strength=10.0)
    return config


def get_layer10_expert5_only_config(intervention_type='force'):
    config = ExpertInterventionConfig()
    if intervention_type == 'force':
        config.force_expert(layer=10, expert_id=5, strength=10.0)
    elif intervention_type == 'suppress':
        config.suppress_expert(layer=10, expert_id=5, strength=-10.0)
    elif intervention_type == 'soft':
        config.soft_bias_expert(layer=10, expert_id=5, strength=2.0)
    return config


def get_layer13_expert1_only_config(intervention_type='force'):
    config = ExpertInterventionConfig()
    if intervention_type == 'force':
        config.force_expert(layer=13, expert_id=1, strength=10.0)
    elif intervention_type == 'suppress':
        config.suppress_expert(layer=13, expert_id=1, strength=-10.0)
    elif intervention_type == 'soft':
        config.soft_bias_expert(layer=13, expert_id=1, strength=2.0)
    return config
