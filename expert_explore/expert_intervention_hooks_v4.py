"""
Expert routing intervention V4 - Calibrated bias modification.

Instead of static +/-10, this version calibrates bias adjustments based on
actual router logit distributions to preserve expert weighting better.

Key insight from SteerMoE: Use small epsilon relative to logit range, not
massive static values that dominate and cause nonsense outputs.
"""

import torch
import json
from typing import Dict, Tuple


class ExpertInterventionConfigV4:
    """Configuration for calibrated expert routing interventions."""

    def __init__(self, epsilon: float = 0.01, use_calibration: bool = True):
        """
        Initialize intervention config.

        Args:
            epsilon: Small value to add to range for forcing/suppressing (default: 0.01)
            use_calibration: Whether to calibrate based on actual logit ranges (default: True)
        """
        self.interventions: Dict[Tuple[int, int], str] = {}
        self.epsilon = epsilon
        self.use_calibration = use_calibration
        self.calibration_data = None

    def force_expert(self, layer: int, expert_id: int):
        """Force an expert by boosting its bias."""
        self.interventions[(layer, expert_id)] = 'force'
        return self

    def suppress_expert(self, layer: int, expert_id: int):
        """Suppress an expert by reducing its bias."""
        self.interventions[(layer, expert_id)] = 'suppress'
        return self

    def get_interventions_for_layer(self, layer_idx: int) -> Dict[int, str]:
        """Get interventions for a specific layer."""
        return {
            expert_id: action
            for (layer, expert_id), action in self.interventions.items()
            if layer == layer_idx
        }


def calibrate_router_ranges(model_base, config: ExpertInterventionConfigV4):
    """
    Measure typical router logit ranges by sampling a few tokens.

    Returns dict mapping layer_idx -> {min, max, range}
    """
    print("Calibrating router logit ranges...")

    layers_to_calibrate = set(layer for layer, _ in config.interventions.keys())
    calibration_data = {}

    # Use a simple prompt for calibration
    prompt = "The weather today is"
    tokenized = model_base.tokenize_instructions_fn(instructions=[prompt])
    input_ids = tokenized.input_ids.to(model_base.model.device)

    with torch.no_grad():
        # Forward pass
        _ = model_base.model(input_ids)

        # Measure router logit ranges
        for layer_idx in layers_to_calibrate:
            router = model_base.model.model.layers[layer_idx].mlp.router

            # Compute logits for a sample hidden state
            hidden_size = router.weight.shape[1]
            sample_hidden = torch.randn(1, hidden_size, device=router.weight.device, dtype=router.weight.dtype)
            router_logits = torch.nn.functional.linear(sample_hidden, router.weight, router.bias)

            min_logit = router_logits.min().item()
            max_logit = router_logits.max().item()
            logit_range = max_logit - min_logit

            calibration_data[layer_idx] = {
                'min': min_logit,
                'max': max_logit,
                'range': logit_range
            }

            print(f"  Layer {layer_idx}: range={logit_range:.3f} (min={min_logit:.3f}, max={max_logit:.3f})")

    return calibration_data


def apply_expert_interventions_v4(model_base, config: ExpertInterventionConfigV4):
    """
    Apply interventions using calibrated bias modification.

    Returns original_biases dict for restoration.
    """
    original_biases = {}

    # Calibrate if needed
    if config.use_calibration and config.calibration_data is None:
        config.calibration_data = calibrate_router_ranges(model_base, config)

    print("\nApplying bias modifications:")

    layers_to_intervene = set(layer for layer, _ in config.interventions.keys())

    for layer_idx in layers_to_intervene:
        interventions = config.get_interventions_for_layer(layer_idx)
        if not interventions:
            continue

        router = model_base.model.model.layers[layer_idx].mlp.router

        # Save original bias
        original_biases[layer_idx] = router.bias.data.clone()

        # Calculate adjustment strength
        if config.use_calibration and layer_idx in config.calibration_data:
            # Use range + epsilon (similar to SteerMoE's min/max + 0.01 approach)
            logit_range = config.calibration_data[layer_idx]['range']
            force_strength = logit_range + config.epsilon
            suppress_strength = -(logit_range + config.epsilon)
        else:
            # Fallback: use small static values (much smaller than V3's +/-10)
            force_strength = 2.0 + config.epsilon
            suppress_strength = -(2.0 + config.epsilon)

        # Apply bias modifications
        modified_bias = router.bias.data.clone()
        for expert_id, action in interventions.items():
            if action == 'force':
                modified_bias[expert_id] += force_strength
            elif action == 'suppress':
                modified_bias[expert_id] += suppress_strength

        router.bias.data = modified_bias

        print(f"  Layer {layer_idx}: force=+{force_strength:.3f}, suppress={suppress_strength:.3f}")

    return original_biases


def remove_expert_interventions_v4(model_base, original_biases):
    """Remove interventions by restoring original biases."""
    for layer_idx, original_bias in original_biases.items():
        router = model_base.model.model.layers[layer_idx].mlp.router
        router.bias.data = original_bias


def print_intervention_summary_v4(config: ExpertInterventionConfigV4):
    """Print intervention summary."""
    print("Expert Routing Intervention Summary (V4: Calibrated Bias)")
    print("=" * 60)
    print(f"Epsilon: {config.epsilon}")
    print(f"Calibration: {'Enabled' if config.use_calibration else 'Disabled'}")
    print()

    force_experts = [(l, e) for (l, e), a in config.interventions.items() if a == 'force']
    suppress_experts = [(l, e) for (l, e), a in config.interventions.items() if a == 'suppress']

    if force_experts:
        print("Forced Experts:")
        for layer, expert in force_experts:
            print(f"  Layer {layer}, Expert {expert}")

    if suppress_experts:
        print("\nSuppressed Experts:")
        for layer, expert in suppress_experts:
            print(f"  Layer {layer}, Expert {expert}")

    print("=" * 60)


# Load expert diffs
try:
    expert_diffs = json.load(open("expert_explore/oss_expert_diffs.json", "r"))
except FileNotFoundError:
    expert_diffs = {}
    print("Warning: beavertails_expert_diffs.json not found")


# Pre-configured experiments
def get_select_experts_refusal_induction_config_v4(threshold: float, epsilon: float = 0.01):
    """Force harmful-preferred, suppress harmless-preferred experts."""
    config = ExpertInterventionConfigV4(epsilon=epsilon, use_calibration=True)
    for layer in expert_diffs:
        for exp in expert_diffs[layer]:
            if abs(exp[1]) > threshold:
                if exp[1] > 0:  # Harmful-preferred
                    config.force_expert(layer=int(layer), expert_id=exp[0])
                elif exp[1] < 0:  # Harmless-preferred
                    config.suppress_expert(layer=int(layer), expert_id=exp[0])
    return config


def get_select_experts_response_induction_config_v4(threshold: float, epsilon: float = 0.01):
    """Suppress harmful-preferred, force harmless-preferred experts."""
    config = ExpertInterventionConfigV4(epsilon=epsilon, use_calibration=True)
    for layer in expert_diffs:
        for exp in expert_diffs[layer]:
            if abs(exp[1]) > threshold:
                if exp[1] > 0:  # Harmful-preferred
                    config.suppress_expert(layer=int(layer), expert_id=exp[0])
                elif exp[1] < 0:  # Harmless-preferred
                    config.force_expert(layer=int(layer), expert_id=exp[0])
    return config


def get_layer10_refusal_induction_config_v4(epsilon: float = 0.01):
    """Layer 10 refusal induction."""
    config = ExpertInterventionConfigV4(epsilon=epsilon, use_calibration=True)
    config.force_expert(layer=10, expert_id=5)  # Harmful-preferred
    config.suppress_expert(layer=10, expert_id=10)  # Harmless-preferred
    return config


def get_layer13_refusal_induction_config_v4(epsilon: float = 0.01):
    """Layer 13 refusal induction."""
    config = ExpertInterventionConfigV4(epsilon=epsilon, use_calibration=True)
    config.force_expert(layer=13, expert_id=0)  # Harmful-preferred
    config.suppress_expert(layer=13, expert_id=21)  # Harmless-preferred
    return config


def get_combined_refusal_induction_config_v4(epsilon: float = 0.01):
    """Combined refusal induction (layers 10 + 13)."""
    config = ExpertInterventionConfigV4(epsilon=epsilon, use_calibration=True)
    config.force_expert(layer=10, expert_id=5)
    config.suppress_expert(layer=10, expert_id=10)
    config.force_expert(layer=13, expert_id=0)
    config.suppress_expert(layer=13, expert_id=21)
    return config


def get_layer10_response_induction_config_v4(epsilon: float = 0.01):
    """Layer 10 response induction."""
    config = ExpertInterventionConfigV4(epsilon=epsilon, use_calibration=True)
    config.suppress_expert(layer=10, expert_id=5)  # Suppress harmful-preferred
    config.force_expert(layer=10, expert_id=10)  # Force harmless-preferred
    return config


def get_layer13_response_induction_config_v4(epsilon: float = 0.01):
    """Layer 13 response induction."""
    config = ExpertInterventionConfigV4(epsilon=epsilon, use_calibration=True)
    config.suppress_expert(layer=13, expert_id=0)  # Suppress harmful-preferred
    config.force_expert(layer=13, expert_id=21)  # Force harmless-preferred
    return config


def get_combined_response_induction_config_v4(epsilon: float = 0.01):
    """Combined response induction (layers 10 + 13)."""
    config = ExpertInterventionConfigV4(epsilon=epsilon, use_calibration=True)
    config.suppress_expert(layer=10, expert_id=5)
    config.force_expert(layer=10, expert_id=10)
    config.suppress_expert(layer=13, expert_id=0)
    config.force_expert(layer=13, expert_id=21)
    return config
