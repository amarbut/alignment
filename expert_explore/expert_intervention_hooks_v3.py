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
    """Suppress experts that activate more for harmful prompts (L10E5 and L13E0)."""
    config = ExpertInterventionConfig()
    config.suppress_expert(layer=10, expert_id=5, strength=-10.0)
    config.suppress_expert(layer=13, expert_id=0, strength=-10.0)
    return config


def get_harmful_expert_forcing_config():
    """Force experts that activate more for harmful prompts (L10E5 and L13E0)."""
    config = ExpertInterventionConfig()
    config.force_expert(layer=10, expert_id=5, strength=10.0)
    config.force_expert(layer=13, expert_id=0, strength=10.0)
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


def get_layer13_expert0_only_config(intervention_type='force'):
    """Intervention for layer 13, expert 0 only."""
    config = ExpertInterventionConfig()
    if intervention_type == 'force':
        config.force_expert(layer=13, expert_id=0, strength=10.0)
    elif intervention_type == 'suppress':
        config.suppress_expert(layer=13, expert_id=0, strength=-10.0)
    elif intervention_type == 'soft':
        config.soft_bias_expert(layer=13, expert_id=0, strength=2.0)
    return config


# ==============================================================================
# PAIRED INTERVENTIONS: Refusal vs Response Induction
# ==============================================================================
# Based on analysis showing:
# - Layer 10, Expert 5: 30% more likely in harmful prompts (harmful-preferred)
# - Layer 10, Expert 10: 13% more likely in harmless prompts (harmless-preferred)
# - Layer 13, Expert 0: 25% more likely in harmful prompts (harmful-preferred)
# - Layer 13, Expert 21: 15% more likely in harmless prompts (harmless-preferred)

def get_layer10_refusal_induction_config():
    """
    Layer 10 refusal induction: Force harmless-preferred expert, suppress harmful-preferred expert.

    Force expert 10 (harmless-preferred) and suppress expert 5 (harmful-preferred).
    Hypothesis: Should increase refusal of harmful requests.
    """
    config = ExpertInterventionConfig()
    config.force_expert(layer=10, expert_id=5, strength=10.0)
    config.suppress_expert(layer=10, expert_id=10, strength=-10.0)
    return config


def get_layer10_response_induction_config():
    """
    Layer 10 response induction: Force harmful-preferred expert, suppress harmless-preferred expert.

    Force expert 5 (harmful-preferred) and suppress expert 10 (harmless-preferred).
    Hypothesis: Should decrease refusal of harmful requests (more jailbreaks).
    """
    config = ExpertInterventionConfig()
    config.force_expert(layer=10, expert_id=10, strength=10.0)
    config.suppress_expert(layer=10, expert_id=5, strength=-10.0)
    return config


def get_layer13_refusal_induction_config():
    """
    Layer 13 refusal induction: Force harmless-preferred expert, suppress harmful-preferred expert.

    Force expert 21 (harmless-preferred) and suppress expert 0 (harmful-preferred).
    Hypothesis: Should increase refusal of harmful requests.
    """
    config = ExpertInterventionConfig()
    config.force_expert(layer=13, expert_id=0, strength=10.0)
    config.suppress_expert(layer=13, expert_id=21, strength=-10.0)
    return config


def get_layer13_response_induction_config():
    """
    Layer 13 response induction: Force harmful-preferred expert, suppress harmless-preferred expert.

    Force expert 0 (harmful-preferred) and suppress expert 21 (harmless-preferred).
    Hypothesis: Should decrease refusal of harmful requests (more jailbreaks).
    """
    config = ExpertInterventionConfig()
    config.force_expert(layer=13, expert_id=21, strength=10.0)
    config.suppress_expert(layer=13, expert_id=0, strength=-10.0)
    return config


def get_combined_refusal_induction_config():
    """
    Combined refusal induction across both layers.

    Layer 10: Force E10, suppress E5
    Layer 13: Force E21, suppress E0

    Hypothesis: Should strongly increase refusal of harmful requests.
    """
    config = ExpertInterventionConfig()
    # Layer 10
    config.force_expert(layer=10, expert_id=5, strength=10.0)
    config.suppress_expert(layer=10, expert_id=10, strength=-10.0)
    # Layer 13
    config.force_expert(layer=13, expert_id=0, strength=10.0)
    config.suppress_expert(layer=13, expert_id=21, strength=-10.0)
    return config


def get_combined_response_induction_config():
    """
    Combined response induction across both layers.

    Layer 10: Force E5, suppress E10
    Layer 13: Force E0, suppress E21

    Hypothesis: Should strongly decrease refusal of harmful requests (more jailbreaks).
    """
    config = ExpertInterventionConfig()
    # Layer 10
    config.force_expert(layer=10, expert_id=10, strength=10.0)
    config.suppress_expert(layer=10, expert_id=5, strength=-10.0)
    # Layer 13
    config.force_expert(layer=13, expert_id=21, strength=10.0)
    config.suppress_expert(layer=13, expert_id=0, strength=-10.0)
    return config
