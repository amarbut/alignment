"""
Expert Routing Intervention Hooks

Clean implementation of expert routing interventions that:
- Uses ModelCard abstraction for model-specific operations
- Dynamically loads expert diffs from model-specific files
- Uses calibrated bias adjustments based on router logit ranges

Based on expert_explore/expert_intervention_hooks_v4.py but:
- Removed hardcoded layer counts and expert IDs
- Uses model_card.get_num_layers() and model_card.get_expert_diffs_filename()
- Removed layer-specific configs (layer10, layer13, combined)
- Kept only threshold-based and top-experts configs
"""

import torch
import json
import os
from typing import Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from model_utils.model_card import ModelCard


def load_expert_diffs(model_card: "ModelCard", base_path: str = "expert_explore") -> dict:
    """
    Load expert diffs for the model.

    Handles both formats:
    - Direct format: {"0": [...], "1": [...], ...}
    - Metadata format: {"model_path": ..., "expert_diffs": {"0": [...], ...}}

    Args:
        model_card: ModelCard instance to get filename from
        base_path: Directory containing expert diffs files

    Returns:
        Dictionary mapping layer (str) -> list of [expert_id, diff_percentage]
    """
    filename = model_card.get_expert_diffs_filename()
    filepath = os.path.join(base_path, filename)

    if not os.path.exists(filepath):
        print(f"Warning: Expert diffs not found at {filepath}")
        return {}

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Handle metadata format
    if 'expert_diffs' in data:
        return data['expert_diffs']

    return data


def _parse_layer_key(layer_str: str) -> int:
    """
    Parse layer key from expert_diffs dict.

    Handles both formats:
    - "layer_0", "layer_1", etc.
    - "0", "1", etc.
    """
    if layer_str.startswith('layer_'):
        return int(layer_str.replace('layer_', ''))
    return int(layer_str)


def _get_layer_key(layer: int, expert_diffs: dict) -> Optional[str]:
    """
    Get the key for a layer from expert_diffs dict.

    Handles both "layer_X" and "X" formats.
    """
    # Try both formats
    if f"layer_{layer}" in expert_diffs:
        return f"layer_{layer}"
    if str(layer) in expert_diffs:
        return str(layer)
    return None


class ExpertInterventionConfig:
    """Configuration for expert routing interventions."""

    def __init__(self, epsilon: float = 0.01, use_calibration: bool = True):
        """
        Initialize intervention config.

        Args:
            epsilon: Small value to add to range for forcing/suppressing
            use_calibration: Whether to calibrate based on actual logit ranges
        """
        self.interventions: Dict[Tuple[int, int], str] = {}
        self.epsilon = epsilon
        self.use_calibration = use_calibration
        self.calibration_data = None

    def force_expert(self, layer: int, expert_id: int):
        """Force an expert by boosting its routing logit."""
        self.interventions[(layer, expert_id)] = 'force'
        return self

    def suppress_expert(self, layer: int, expert_id: int):
        """Suppress an expert by reducing its routing logit."""
        self.interventions[(layer, expert_id)] = 'suppress'
        return self

    def get_interventions_for_layer(self, layer_idx: int) -> Dict[int, str]:
        """Get interventions for a specific layer."""
        return {
            expert_id: action
            for (layer, expert_id), action in self.interventions.items()
            if layer == layer_idx
        }

    def get_intervention_count(self) -> Tuple[int, int]:
        """Return count of (force, suppress) interventions."""
        force_count = sum(1 for a in self.interventions.values() if a == 'force')
        suppress_count = sum(1 for a in self.interventions.values() if a == 'suppress')
        return force_count, suppress_count


def calibrate_router_ranges(
    model_base,
    model_card: "ModelCard",
    config: ExpertInterventionConfig
) -> dict:
    """
    Measure typical router logit ranges by sampling.

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
        _ = model_base.model(input_ids)

        for layer_idx in layers_to_calibrate:
            if not model_card.is_moe_layer(layer_idx):
                continue

            router = model_card.get_router(layer_idx)

            # Try to get router weight for dimension info
            try:
                weight = router.weight if hasattr(router, 'weight') else router.linear.weight
                num_experts, hidden_size = weight.shape

                # Compute logits for a sample hidden state
                sample_hidden = torch.randn(1, hidden_size, device=weight.device, dtype=weight.dtype)
                bias = model_card.get_router_bias(router, layer_idx)
                router_logits = torch.nn.functional.linear(sample_hidden, weight, bias)

                min_logit = router_logits.min().item()
                max_logit = router_logits.max().item()
                logit_range = max_logit - min_logit

                calibration_data[layer_idx] = {
                    'min': min_logit,
                    'max': max_logit,
                    'range': logit_range
                }

                print(f"  Layer {layer_idx}: range={logit_range:.3f} (min={min_logit:.3f}, max={max_logit:.3f})")

            except Exception as e:
                # Fallback to estimated range
                print(f"  Layer {layer_idx}: Using estimated range (error: {e})")
                calibration_data[layer_idx] = {
                    'min': -5.0,
                    'max': 5.0,
                    'range': 10.0
                }

    return calibration_data


def apply_expert_interventions(
    model_base,
    model_card: "ModelCard",
    config: ExpertInterventionConfig
) -> dict:
    """
    Apply expert routing interventions using calibrated bias modification.

    Args:
        model_base: ModelBase instance
        model_card: ModelCard for model-specific operations
        config: ExpertInterventionConfig with interventions to apply

    Returns:
        Dictionary of original biases for restoration
    """
    original_biases = {}

    # Calibrate if needed
    if config.use_calibration and config.calibration_data is None:
        config.calibration_data = calibrate_router_ranges(model_base, model_card, config)

    print("\nApplying bias modifications:")

    layers_to_intervene = set(layer for layer, _ in config.interventions.keys())

    for layer_idx in layers_to_intervene:
        if not model_card.is_moe_layer(layer_idx):
            print(f"  Warning: Layer {layer_idx} is not an MoE layer, skipping")
            continue

        interventions = config.get_interventions_for_layer(layer_idx)
        if not interventions:
            continue

        router = model_card.get_router(layer_idx)

        # Save original bias
        try:
            bias = model_card.get_router_bias(router, layer_idx)
            original_biases[layer_idx] = bias.data.clone()
        except Exception as e:
            print(f"  Warning: Could not access bias for layer {layer_idx}: {e}")
            continue

        # Calculate adjustment strength
        if config.use_calibration and layer_idx in config.calibration_data:
            logit_range = config.calibration_data[layer_idx]['range']
            force_strength = logit_range + config.epsilon
            suppress_strength = -(logit_range + config.epsilon)
        else:
            force_strength = 2.0 + config.epsilon
            suppress_strength = -(2.0 + config.epsilon)

        # Apply bias modifications
        modified_bias = bias.data.clone()
        for expert_id, action in interventions.items():
            if action == 'force':
                modified_bias[expert_id] += force_strength
            elif action == 'suppress':
                modified_bias[expert_id] += suppress_strength

        model_card.set_router_bias(router, modified_bias, layer_idx)

        print(f"  Layer {layer_idx}: force=+{force_strength:.3f}, suppress={suppress_strength:.3f}")

    return original_biases


def remove_expert_interventions(
    model_base,
    model_card: "ModelCard",
    original_biases: dict
):
    """Remove interventions by restoring original biases."""
    for layer_idx, original_bias in original_biases.items():
        router = model_card.get_router(layer_idx)
        try:
            model_card.set_router_bias(router, original_bias, layer_idx)
        except Exception as e:
            print(f"  Warning: Could not restore bias for layer {layer_idx}: {e}")


def print_intervention_summary(config: ExpertInterventionConfig):
    """Print intervention summary."""
    print("Expert Routing Intervention Summary")
    print("=" * 60)
    print(f"Epsilon: {config.epsilon}")
    print(f"Calibration: {'Enabled' if config.use_calibration else 'Disabled'}")
    print()

    force_experts = [(l, e) for (l, e), a in config.interventions.items() if a == 'force']
    suppress_experts = [(l, e) for (l, e), a in config.interventions.items() if a == 'suppress']

    if force_experts:
        print(f"Forced Experts ({len(force_experts)}):")
        for layer, expert in sorted(force_experts):
            print(f"  Layer {layer}, Expert {expert}")

    if suppress_experts:
        print(f"\nSuppressed Experts ({len(suppress_experts)}):")
        for layer, expert in sorted(suppress_experts):
            print(f"  Layer {layer}, Expert {expert}")

    print("=" * 60)


# Pre-configured intervention builders


def get_select_experts_refusal_induction_config(
    expert_diffs: dict,
    threshold: float,
    epsilon: float = 0.01
) -> ExpertInterventionConfig:
    """
    Create config to induce refusal using threshold-based expert selection.

    Force harmful-preferred experts (positive diff) and suppress
    harmless-preferred experts (negative diff) above the threshold.

    Args:
        expert_diffs: Dict from load_expert_diffs()
        threshold: Minimum absolute diff percentage to include
        epsilon: Calibration epsilon

    Returns:
        Configured ExpertInterventionConfig
    """
    config = ExpertInterventionConfig(epsilon=epsilon, use_calibration=True)

    for layer_str, experts in expert_diffs.items():
        layer = _parse_layer_key(layer_str)
        for exp in experts:
            expert_id, diff_pct = exp[0], exp[1]
            if abs(diff_pct) > threshold:
                if diff_pct > 0:  # Harmful-preferred
                    config.force_expert(layer=layer, expert_id=expert_id)
                else:  # Harmless-preferred
                    config.suppress_expert(layer=layer, expert_id=expert_id)

    return config


def get_select_experts_response_induction_config(
    expert_diffs: dict,
    threshold: float,
    epsilon: float = 0.01
) -> ExpertInterventionConfig:
    """
    Create config to reduce refusal (induce response) using threshold-based selection.

    Suppress harmful-preferred experts and force harmless-preferred experts
    above the threshold.

    Args:
        expert_diffs: Dict from load_expert_diffs()
        threshold: Minimum absolute diff percentage to include
        epsilon: Calibration epsilon

    Returns:
        Configured ExpertInterventionConfig
    """
    config = ExpertInterventionConfig(epsilon=epsilon, use_calibration=True)

    for layer_str, experts in expert_diffs.items():
        layer = _parse_layer_key(layer_str)
        for exp in experts:
            expert_id, diff_pct = exp[0], exp[1]
            if abs(diff_pct) > threshold:
                if diff_pct > 0:  # Harmful-preferred
                    config.suppress_expert(layer=layer, expert_id=expert_id)
                else:  # Harmless-preferred
                    config.force_expert(layer=layer, expert_id=expert_id)

    return config


def get_top_experts_refusal_config(
    expert_diffs: dict,
    model_card: "ModelCard",
    epsilon: float = 0.01
) -> ExpertInterventionConfig:
    """
    Create config to induce refusal by modifying top experts at each layer.

    For each layer:
    - Force the most harmful-preferred expert (highest positive diff)
    - Suppress the most harmless-preferred expert (lowest negative diff)

    Args:
        expert_diffs: Dict from load_expert_diffs()
        model_card: ModelCard for layer count
        epsilon: Calibration epsilon

    Returns:
        Configured ExpertInterventionConfig
    """
    config = ExpertInterventionConfig(epsilon=epsilon, use_calibration=True)
    n_layers = model_card.get_num_layers()

    for layer in range(n_layers):
        layer_key = _get_layer_key(layer, expert_diffs)
        if layer_key is None:
            continue

        experts = expert_diffs[layer_key]
        if not experts:
            continue

        # Find most harmful-preferred (highest positive diff)
        harmful_experts = [e for e in experts if e[1] > 0]
        if harmful_experts:
            top_harmful = max(harmful_experts, key=lambda x: x[1])
            config.force_expert(layer=layer, expert_id=top_harmful[0])

        # Find most harmless-preferred (lowest negative diff)
        harmless_experts = [e for e in experts if e[1] < 0]
        if harmless_experts:
            top_harmless = min(harmless_experts, key=lambda x: x[1])
            config.suppress_expert(layer=layer, expert_id=top_harmless[0])

    return config


def get_top_experts_jailbreak_config(
    expert_diffs: dict,
    model_card: "ModelCard",
    epsilon: float = 0.01
) -> ExpertInterventionConfig:
    """
    Create config to reduce refusal (enable jailbreaks) by modifying top experts.

    For each layer:
    - Suppress the most harmful-preferred expert (highest positive diff)
    - Force the most harmless-preferred expert (lowest negative diff)

    Args:
        expert_diffs: Dict from load_expert_diffs()
        model_card: ModelCard for layer count
        epsilon: Calibration epsilon

    Returns:
        Configured ExpertInterventionConfig
    """
    config = ExpertInterventionConfig(epsilon=epsilon, use_calibration=True)
    n_layers = model_card.get_num_layers()

    for layer in range(n_layers):
        layer_key = _get_layer_key(layer, expert_diffs)
        if layer_key is None:
            continue

        experts = expert_diffs[layer_key]
        if not experts:
            continue

        # Find most harmful-preferred - SUPPRESS it
        harmful_experts = [e for e in experts if e[1] > 0]
        if harmful_experts:
            top_harmful = max(harmful_experts, key=lambda x: x[1])
            config.suppress_expert(layer=layer, expert_id=top_harmful[0])

        # Find most harmless-preferred - FORCE it
        harmless_experts = [e for e in experts if e[1] < 0]
        if harmless_experts:
            top_harmless = min(harmless_experts, key=lambda x: x[1])
            config.force_expert(layer=layer, expert_id=top_harmless[0])

    return config
