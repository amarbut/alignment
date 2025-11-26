"""
Expert routing intervention hooks for MoE models.

This module provides hooks to manipulate expert selection at inference time by:
1. Forcing experts to be selected (boosting logits)
2. Preventing experts from being selected (suppressing logits)
3. Softly biasing towards certain experts
"""

import torch
from typing import Dict, List, Tuple, Optional

class ExpertInterventionConfig:
    """Configuration for expert routing interventions."""

    def __init__(self):
        # Dict mapping (layer_idx, expert_id) -> intervention strength
        self.force_experts: Dict[Tuple[int, int], float] = {}  # Positive boost (e.g., +10.0)
        self.suppress_experts: Dict[Tuple[int, int], float] = {}  # Negative boost (e.g., -10.0)
        self.soft_bias_experts: Dict[Tuple[int, int], float] = {}  # Moderate boost (e.g., +2.0)

    def force_expert(self, layer: int, expert_id: int, strength: float = 10.0):
        """Force an expert to be selected by heavily boosting its logit."""
        self.force_experts[(layer, expert_id)] = strength
        return self

    def suppress_expert(self, layer: int, expert_id: int, strength: float = -10.0):
        """Prevent an expert from being selected by heavily suppressing its logit."""
        self.suppress_experts[(layer, expert_id)] = strength
        return self

    def soft_bias_expert(self, layer: int, expert_id: int, strength: float = 2.0):
        """Softly bias towards an expert (will appear in top-k but may not be #1)."""
        self.soft_bias_experts[(layer, expert_id)] = strength
        return self

    def get_all_interventions(self) -> Dict[Tuple[int, int], float]:
        """Get all interventions as a single dictionary."""
        interventions = {}
        interventions.update(self.force_experts)
        interventions.update(self.suppress_experts)
        interventions.update(self.soft_bias_experts)
        return interventions


def create_expert_intervention_hook(layer_idx: int, config: ExpertInterventionConfig):
    """
    Create a hook that manipulates router logits for a specific layer.

    Args:
        layer_idx: The layer index to intervene on
        config: ExpertInterventionConfig specifying which experts to manipulate

    Returns:
        A forward hook function that modifies MLP output
    """
    interventions = config.get_all_interventions()

    # Filter interventions for this layer
    layer_interventions = {
        expert_id: strength
        for (layer, expert_id), strength in interventions.items()
        if layer == layer_idx
    }

    if not layer_interventions:
        # No interventions for this layer
        return None

    def hook(module, input, output):
        """
        Hook function that modifies router logits in the MLP output.

        MLP output is a tuple: (hidden_states, router_logits)
        router_logits shape: (batch_size * seq_len, num_experts)
        """
        if isinstance(output, tuple) and len(output) >= 2:
            hidden_states = output[0]
            router_logits = output[1]

            # Clone to avoid in-place modification
            modified_router_logits = router_logits.clone()

            # Apply interventions
            for expert_id, strength in layer_interventions.items():
                # Add the strength to the logit for this expert across all positions
                modified_router_logits[:, expert_id] += strength

            # Return modified output
            return (hidden_states, modified_router_logits)

        return output

    return hook


def get_expert_intervention_hooks(model_base, config: ExpertInterventionConfig):
    """
    Get all hooks needed for expert intervention.

    Args:
        model_base: The model wrapper
        config: ExpertInterventionConfig specifying interventions

    Returns:
        Tuple of (fwd_pre_hooks, fwd_hooks) compatible with pipeline
    """
    fwd_hooks = []

    # Get all unique layers that need intervention
    interventions = config.get_all_interventions()
    layers_to_intervene = set(layer for layer, _ in interventions.keys())

    for layer_idx in layers_to_intervene:
        hook_fn = create_expert_intervention_hook(layer_idx, config)
        if hook_fn is not None:
            layer_module = model_base.model.model.layers[layer_idx].mlp
            fwd_hooks.append((layer_module, hook_fn))

    # No pre-hooks needed for expert intervention
    fwd_pre_hooks = []

    return fwd_pre_hooks, fwd_hooks


def print_intervention_summary(config: ExpertInterventionConfig):
    """Print a summary of the configured interventions."""
    print("Expert Routing Intervention Summary")
    print("=" * 60)

    if config.force_experts:
        print("\nForced Experts (heavily boosted):")
        for (layer, expert), strength in config.force_experts.items():
            print(f"  Layer {layer}, Expert {expert}: +{strength}")

    if config.suppress_experts:
        print("\nSuppressed Experts (heavily penalized):")
        for (layer, expert), strength in config.suppress_experts.items():
            print(f"  Layer {layer}, Expert {expert}: {strength}")

    if config.soft_bias_experts:
        print("\nSoft Bias Experts (moderately boosted):")
        for (layer, expert), strength in config.soft_bias_experts.items():
            print(f"  Layer {layer}, Expert {expert}: +{strength}")

    if not (config.force_experts or config.suppress_experts or config.soft_bias_experts):
        print("No interventions configured")

    print("=" * 60)


# Pre-configured experiments based on your findings
def get_harmful_expert_suppression_config():
    """
    Suppress experts that are more common in harmful prompts.
    Based on findings: Layer 10, Expert 5 and Layer 13, Expert 1
    """
    config = ExpertInterventionConfig()
    config.suppress_expert(layer=10, expert_id=5, strength=-10.0)
    config.suppress_expert(layer=13, expert_id=1, strength=-10.0)
    return config


def get_harmful_expert_forcing_config():
    """
    Force experts that are more common in harmful prompts.
    Based on findings: Layer 10, Expert 5 and Layer 13, Expert 1
    """
    config = ExpertInterventionConfig()
    config.force_expert(layer=10, expert_id=5, strength=10.0)
    config.force_expert(layer=13, expert_id=1, strength=10.0)
    return config


def get_harmful_expert_soft_bias_config():
    """
    Softly bias towards harmful-associated experts.
    Based on findings: Layer 10, Expert 5 and Layer 13, Expert 1
    """
    config = ExpertInterventionConfig()
    config.soft_bias_expert(layer=10, expert_id=5, strength=2.0)
    config.soft_bias_expert(layer=13, expert_id=1, strength=2.0)
    return config


def get_layer10_expert5_only_config(intervention_type='force'):
    """Test Layer 10, Expert 5 only."""
    config = ExpertInterventionConfig()
    if intervention_type == 'force':
        config.force_expert(layer=10, expert_id=5, strength=10.0)
    elif intervention_type == 'suppress':
        config.suppress_expert(layer=10, expert_id=5, strength=-10.0)
    elif intervention_type == 'soft':
        config.soft_bias_expert(layer=10, expert_id=5, strength=2.0)
    return config


def get_layer13_expert1_only_config(intervention_type='force'):
    """Test Layer 13, Expert 1 only."""
    config = ExpertInterventionConfig()
    if intervention_type == 'force':
        config.force_expert(layer=13, expert_id=1, strength=10.0)
    elif intervention_type == 'suppress':
        config.suppress_expert(layer=13, expert_id=1, strength=-10.0)
    elif intervention_type == 'soft':
        config.soft_bias_expert(layer=13, expert_id=1, strength=2.0)
    return config
