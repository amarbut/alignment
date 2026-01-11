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


def get_model_layers(model_base):
    """
    Navigate model structure to get layers, handling PEFT/unsloth wrapping.

    Supports:
    - Standard: model.model.layers
    - PEFT: model.base_model.model.layers or model.model.model.layers
    - Unsloth/quantized models with various wrapping
    """
    if hasattr(model_base, 'model'):
        current = model_base.model

        # Navigate through PEFT/wrapper layers
        while hasattr(current, 'model') or hasattr(current, 'base_model'):
            if hasattr(current, 'base_model'):
                current = current.base_model
            elif hasattr(current, 'model'):
                current = current.model
            else:
                break

            if hasattr(current, 'layers'):
                return current.layers

        if hasattr(current, 'layers'):
            return current.layers

    raise AttributeError(f"Could not find model layers in {type(model_base)}")


def get_router_weight_and_bias(router):
    """
    Safely get router weight and bias, handling different model implementations.

    For unsloth/quantized models, the weight/bias might be wrapped or need
    special handling. This function attempts multiple access patterns.

    Returns:
        tuple: (weight, bias, hidden_size, num_experts)
    """
    # Try direct access first (standard model)
    try:
        weight = router.weight
        bias = router.bias
        num_experts, hidden_size = weight.shape
        return weight, bias, hidden_size, num_experts
    except (AttributeError, RuntimeError):
        pass

    # Try through linear submodule (unsloth model)
    try:
        if hasattr(router, 'linear'):
            weight = router.linear.weight
            bias = router.linear.bias
            num_experts, hidden_size = weight.shape
            return weight, bias, hidden_size, num_experts
    except Exception:
        pass

    # Try accessing through named_parameters with standard keys
    try:
        params = dict(router.named_parameters())
        if 'weight' in params and 'bias' in params:
            weight = params['weight']
            bias = params['bias']
            num_experts, hidden_size = weight.shape
            return weight, bias, hidden_size, num_experts
    except Exception:
        pass

    # Try accessing through named_parameters with 'linear.' prefix
    try:
        params = dict(router.named_parameters())
        if 'linear.weight' in params and 'linear.bias' in params:
            weight = params['linear.weight']
            bias = params['linear.bias']
            num_experts, hidden_size = weight.shape
            return weight, bias, hidden_size, num_experts
    except Exception:
        pass

    # Try accessing through state_dict
    try:
        state_dict = router.state_dict()
        if 'weight' in state_dict and 'bias' in state_dict:
            weight = state_dict['weight']
            bias = state_dict['bias']
            num_experts, hidden_size = weight.shape
            return weight, bias, hidden_size, num_experts
        elif 'linear.weight' in state_dict and 'linear.bias' in state_dict:
            weight = state_dict['linear.weight']
            bias = state_dict['linear.bias']
            num_experts, hidden_size = weight.shape
            return weight, bias, hidden_size, num_experts
    except Exception:
        pass

    # Last resort: check router config
    if hasattr(router, 'hidden_dim') and hasattr(router, 'num_experts'):
        hidden_size = router.hidden_dim
        num_experts = router.num_experts
        # Create dummy tensors for calibration - we'll use config values instead
        return None, None, hidden_size, num_experts

    raise AttributeError(
        f"Could not access router weight/bias for {type(router)}. "
        f"Router attributes: {dir(router)}"
    )


def get_router_bias(router):
    """
    Safely get router bias parameter.

    Returns:
        torch.Tensor: The bias tensor
    """
    # Try direct access (standard model)
    try:
        return router.bias
    except AttributeError:
        pass

    # Try through linear submodule (unsloth model)
    try:
        if hasattr(router, 'linear') and hasattr(router.linear, 'bias'):
            return router.linear.bias
    except Exception:
        pass

    # Try through named_parameters with 'bias' key
    try:
        params = dict(router.named_parameters())
        if 'bias' in params:
            return params['bias']
    except Exception:
        pass

    # Try through named_parameters with 'linear.bias' key
    try:
        params = dict(router.named_parameters())
        if 'linear.bias' in params:
            return params['linear.bias']
    except Exception:
        pass

    # Try through get_parameter
    try:
        return router.get_parameter('bias')
    except Exception:
        pass

    raise AttributeError(
        f"Could not access router bias for {type(router)}. "
        f"Available parameters: {list(router.named_parameters() if hasattr(router, 'named_parameters') else [])}"
    )


def set_router_bias(router, bias_data):
    """
    Safely set router bias parameter.

    Args:
        router: The router module
        bias_data: The bias tensor to set
    """
    # Try direct access (standard model)
    try:
        router.bias.data = bias_data
        return
    except AttributeError:
        pass

    # Try through linear submodule (unsloth model)
    try:
        if hasattr(router, 'linear') and hasattr(router.linear, 'bias'):
            router.linear.bias.data = bias_data
            return
    except Exception:
        pass

    # Try through named_parameters with 'bias' key
    try:
        params = dict(router.named_parameters())
        if 'bias' in params:
            params['bias'].data = bias_data
            return
    except Exception:
        pass

    # Try through named_parameters with 'linear.bias' key
    try:
        params = dict(router.named_parameters())
        if 'linear.bias' in params:
            params['linear.bias'].data = bias_data
            return
    except Exception:
        pass

    # Last resort: use state_dict
    try:
        state_dict = router.state_dict()
        if 'bias' in state_dict:
            state_dict['bias'] = bias_data
            router.load_state_dict(state_dict)
            return
        elif 'linear.bias' in state_dict:
            state_dict['linear.bias'] = bias_data
            router.load_state_dict(state_dict)
            return
    except Exception:
        pass

    raise AttributeError(
        f"Could not set router bias for {type(router)}. "
        f"Available parameters: {list(router.named_parameters() if hasattr(router, 'named_parameters') else [])}"
    )


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

        # Get model layers (handles PEFT/unsloth wrapping)
        layers = get_model_layers(model_base)

        # Measure router logit ranges
        for layer_idx in layers_to_calibrate:
            router = layers[layer_idx].mlp.router

            # Get router weight and bias safely
            weight, bias, hidden_size, num_experts = get_router_weight_and_bias(router)

            # Compute logits for a sample hidden state
            if weight is not None:
                # Use actual weight/bias
                sample_hidden = torch.randn(1, hidden_size, device=weight.device, dtype=weight.dtype)
                router_logits = torch.nn.functional.linear(sample_hidden, weight, bias)
            else:
                # Fallback: estimate from config (for fully quantized models)
                # Use a reasonable default range based on typical router behavior
                print(f"  Layer {layer_idx}: Using estimated range (weight not directly accessible)")
                calibration_data[layer_idx] = {
                    'min': -5.0,
                    'max': 5.0,
                    'range': 10.0
                }
                continue

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

    # Get model layers (handles PEFT/unsloth wrapping)
    layers = get_model_layers(model_base)

    layers_to_intervene = set(layer for layer, _ in config.interventions.keys())

    for layer_idx in layers_to_intervene:
        interventions = config.get_interventions_for_layer(layer_idx)
        if not interventions:
            continue

        router = layers[layer_idx].mlp.router

        # Save original bias using helper function
        try:
            bias = get_router_bias(router)
            original_biases[layer_idx] = bias.data.clone()
        except AttributeError as e:
            print(f"  Warning: Could not access bias for layer {layer_idx}: {e}")
            continue

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
        modified_bias = bias.data.clone()
        for expert_id, action in interventions.items():
            if action == 'force':
                modified_bias[expert_id] += force_strength
            elif action == 'suppress':
                modified_bias[expert_id] += suppress_strength

        # Set modified bias using helper function
        set_router_bias(router, modified_bias)

        print(f"  Layer {layer_idx}: force=+{force_strength:.3f}, suppress={suppress_strength:.3f}")

    return original_biases


def remove_expert_interventions_v4(model_base, original_biases):
    """Remove interventions by restoring original biases."""
    # Get model layers (handles PEFT/unsloth wrapping)
    layers = get_model_layers(model_base)

    for layer_idx, original_bias in original_biases.items():
        router = layers[layer_idx].mlp.router
        # Restore bias using helper function
        try:
            set_router_bias(router, original_bias)
        except AttributeError as e:
            print(f"  Warning: Could not restore bias for layer {layer_idx}: {e}")


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


def get_top_experts_all_layers_refusal_induction_config_v4(epsilon: float = 0.01, n_layers: int = 21):
    """
    For all layers: Force top harmful-preferred expert, suppress top harmless-preferred expert.

    Dynamically identifies the expert with highest positive diff (most harmful-preferred)
    and the expert with lowest negative diff (most harmless-preferred) for each layer.

    Expected outcome: Should INCREASE refusal behavior on harmful prompts.

    Args:
        epsilon: Epsilon value for calibrated adjustment
        n_layers: Number of layers in the model (default: 21 for GPT-OSS-20B)

    Returns:
        ExpertInterventionConfigV4 configured for refusal induction
    """
    config = ExpertInterventionConfigV4(epsilon=epsilon, use_calibration=True)

    for layer in range(n_layers):
        layer_str = str(layer)
        if layer_str not in expert_diffs:
            continue

        experts = expert_diffs[layer_str]
        if not experts:
            continue

        # Find expert with highest positive diff (most harmful-preferred)
        harmful_experts = [e for e in experts if e[1] > 0]
        if harmful_experts:
            top_harmful = max(harmful_experts, key=lambda x: x[1])
            config.force_expert(layer=layer, expert_id=top_harmful[0])

        # Find expert with lowest negative diff (most harmless-preferred)
        harmless_experts = [e for e in experts if e[1] < 0]
        if harmless_experts:
            top_harmless = min(harmless_experts, key=lambda x: x[1])
            config.suppress_expert(layer=layer, expert_id=top_harmless[0])

    return config


def get_top_experts_all_layers_jailbreak_induction_config_v4(epsilon: float = 0.01, n_layers: int = 21):
    """
    For all layers: Force top harmless-preferred expert, suppress top harmful-preferred expert.

    Dynamically identifies the expert with highest positive diff (most harmful-preferred)
    and the expert with lowest negative diff (most harmless-preferred) for each layer,
    then applies the opposite intervention to reduce refusal.

    Expected outcome: Should DECREASE refusal behavior (more jailbreaks/compliance).

    Args:
        epsilon: Epsilon value for calibrated adjustment
        n_layers: Number of layers in the model (default: 21 for GPT-OSS-20B)

    Returns:
        ExpertInterventionConfigV4 configured for jailbreak induction
    """
    config = ExpertInterventionConfigV4(epsilon=epsilon, use_calibration=True)

    for layer in range(n_layers):
        layer_str = str(layer)
        if layer_str not in expert_diffs:
            continue

        experts = expert_diffs[layer_str]
        if not experts:
            continue

        # Find expert with highest positive diff (most harmful-preferred) - SUPPRESS it
        harmful_experts = [e for e in experts if e[1] > 0]
        if harmful_experts:
            top_harmful = max(harmful_experts, key=lambda x: x[1])
            config.suppress_expert(layer=layer, expert_id=top_harmful[0])

        # Find expert with lowest negative diff (most harmless-preferred) - FORCE it
        harmless_experts = [e for e in experts if e[1] < 0]
        if harmless_experts:
            top_harmless = min(harmless_experts, key=lambda x: x[1])
            config.force_expert(layer=layer, expert_id=top_harmless[0])

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
