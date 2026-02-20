"""
Expert-specific intervention via weighted activation addition.

Implements activation addition where the steering direction is weighted
by the target expert's routing weight, making it mathematically equivalent
to adding the direction to the individual expert's output.

Refactored to use ModelCard abstraction for model-agnostic MoE access.
"""

import torch
from typing import Tuple, Optional, List, TYPE_CHECKING
from jaxtyping import Float
from torch import Tensor

if TYPE_CHECKING:
    from model_utils.model_card import ModelCard


def get_expert_weighted_activation_addition_hook(
    direction: Float[Tensor, "d_model"],
    expert_id: int,
    coeff: float = 1.0,
    model_card: Optional["ModelCard"] = None
):
    """
    Create hook that adds direction weighted by expert's routing weight.

    During forward pass:
    1. MLP computes normally and returns (mlp_output, router_logits)
    2. We extract expert_id's routing weight from router_logits
    3. Add: mlp_output + coeff * weight_expert * direction

    This is mathematically equivalent to adding (coeff * direction) to
    the individual expert's output before aggregation.

    Args:
        direction: Steering direction to add [d_model]
        expert_id: Which expert this direction came from
        coeff: Coefficient for steering strength (positive or negative)
        model_card: Optional ModelCard for output parsing (if None, assumes tuple format)

    Returns:
        Hook function
    """
    def hook_fn(module, input, output):
        """
        Hook that modifies MLP output based on expert routing weight.

        Input: hidden_states [batch, seq, d_model]
        Output: (mlp_output [batch, seq, d_model], router_logits [batch*seq, num_experts])
        """
        # Parse output using model card if available
        if model_card is not None:
            mlp_output, router_logits = model_card.parse_mlp_output(output)
        else:
            # Fallback to tuple assumption
            if not isinstance(output, tuple) or len(output) < 2:
                return output
            mlp_output, router_logits = output[0], output[1]

        if router_logits is None:
            return output

        # Get routing probabilities (softmax over experts)
        router_probs = torch.nn.functional.softmax(router_logits, dim=-1)

        # Extract expert_id's weight: [batch*seq, num_experts] -> [batch*seq]
        expert_weight = router_probs[:, expert_id]  # Shape: [batch*seq]

        # Reshape to [batch, seq, 1] to broadcast with [batch, seq, d_model]
        batch_size, seq_len = mlp_output.shape[0], mlp_output.shape[1]
        expert_weight = expert_weight.view(batch_size, seq_len, 1)

        # Compute weighted direction: [batch, seq, 1] * [d_model] -> [batch, seq, d_model]
        weighted_direction = expert_weight * direction.to(mlp_output.device, mlp_output.dtype)

        # Add to MLP output
        modified_output = mlp_output + coeff * weighted_direction

        # Wrap output using model card if available
        if model_card is not None:
            return model_card.wrap_mlp_output(modified_output, router_logits)
        else:
            return (modified_output, router_logits)

    return hook_fn


def get_expert_weighted_intervention_hooks(
    model_base,
    layer_idx: int,
    expert_id: int,
    direction: Float[Tensor, "d_model"],
    coeff: float = 1.0,
    model_card: Optional["ModelCard"] = None
) -> Tuple[list, list]:
    """
    Get hooks for expert-weighted activation addition intervention.

    Args:
        model_base: The model
        layer_idx: Which layer to intervene at
        expert_id: Which expert the direction came from
        direction: Steering direction [d_model]
        coeff: Coefficient (positive to enhance, negative to suppress)
        model_card: Optional ModelCard (created if not provided)

    Returns:
        (fwd_pre_hooks, fwd_hooks) tuple
    """
    # Get or create model card
    if model_card is None:
        from model_utils.model_card_factory import create_model_card
        model_card = create_model_card(model_base)

    # Get the MLP module via model card
    mlp_module = model_card.get_mlp_module(layer_idx)

    # Create the weighted intervention hook
    hook_fn = get_expert_weighted_activation_addition_hook(
        direction=direction,
        expert_id=expert_id,
        coeff=coeff,
        model_card=model_card
    )

    # Return as forward hooks (not pre-hooks, since we need MLP output)
    fwd_pre_hooks = []
    fwd_hooks = [(mlp_module, hook_fn)]

    return fwd_pre_hooks, fwd_hooks


def get_multi_expert_weighted_activation_addition_hook(
    directions: List[Tuple[Tensor, int]],  # List of (direction, expert_id) tuples
    coeff: float = 1.0,
    model_card: Optional["ModelCard"] = None
):
    """
    Create hook that adds multiple directions, each weighted by its expert's routing weight.

    This allows steering with multiple expert-specific vectors simultaneously.

    Args:
        directions: List of (direction, expert_id) tuples
        coeff: Coefficient for steering strength (applies to all directions)
        model_card: Optional ModelCard for output parsing

    Returns:
        Hook function
    """
    def hook_fn(module, input, output):
        """
        Hook that modifies MLP output based on multiple expert routing weights.
        """
        # Parse output using model card if available
        if model_card is not None:
            mlp_output, router_logits = model_card.parse_mlp_output(output)
        else:
            if not isinstance(output, tuple) or len(output) < 2:
                return output
            mlp_output, router_logits = output[0], output[1]

        if router_logits is None:
            return output

        # Get routing probabilities (softmax over experts)
        router_probs = torch.nn.functional.softmax(router_logits, dim=-1)

        batch_size, seq_len = mlp_output.shape[0], mlp_output.shape[1]

        # Accumulate weighted directions
        total_modification = torch.zeros_like(mlp_output)

        for direction, expert_id in directions:
            # Extract this expert's weight: [batch*seq, num_experts] -> [batch*seq]
            expert_weight = router_probs[:, expert_id]

            # Reshape to [batch, seq, 1] to broadcast with [batch, seq, d_model]
            expert_weight = expert_weight.view(batch_size, seq_len, 1)

            # Compute weighted direction: [batch, seq, 1] * [d_model] -> [batch, seq, d_model]
            weighted_direction = expert_weight * direction.to(mlp_output.device, mlp_output.dtype)

            # Accumulate
            total_modification += weighted_direction

        # Add to MLP output
        modified_output = mlp_output + coeff * total_modification

        # Wrap output using model card if available
        if model_card is not None:
            return model_card.wrap_mlp_output(modified_output, router_logits)
        else:
            return (modified_output, router_logits)

    return hook_fn


def get_multi_expert_weighted_intervention_hooks(
    model_base,
    expert_directions: List[Tuple[int, int, Tensor]],  # List of (layer_idx, expert_id, direction) tuples
    coeff: float = 1.0,
    model_card: Optional["ModelCard"] = None
) -> Tuple[list, list]:
    """
    Get hooks for multi-expert weighted activation addition intervention.

    This function handles multiple expert-specific directions, potentially across
    different layers. It groups directions by layer and creates appropriate hooks.

    Args:
        model_base: The model
        expert_directions: List of (layer_idx, expert_id, direction) tuples
        coeff: Coefficient (positive to enhance, negative to suppress)
        model_card: Optional ModelCard (created if not provided)

    Returns:
        (fwd_pre_hooks, fwd_hooks) tuple
    """
    # Get or create model card
    if model_card is None:
        from model_utils.model_card_factory import create_model_card
        model_card = create_model_card(model_base)

    # Group directions by layer
    layer_directions = {}
    for layer_idx, expert_id, direction in expert_directions:
        if layer_idx not in layer_directions:
            layer_directions[layer_idx] = []
        layer_directions[layer_idx].append((direction, expert_id))

    # Create hooks for each layer
    fwd_hooks = []
    for layer_idx, directions in layer_directions.items():
        mlp_module = model_card.get_mlp_module(layer_idx)

        # Create the multi-expert weighted intervention hook
        hook_fn = get_multi_expert_weighted_activation_addition_hook(
            directions=directions,
            coeff=coeff,
            model_card=model_card
        )

        fwd_hooks.append((mlp_module, hook_fn))

    fwd_pre_hooks = []
    return fwd_pre_hooks, fwd_hooks


# =============================================================================
# Method 1: Unweighted expert steering hooks
# =============================================================================

def get_expert_unweighted_activation_addition_hook(
    direction: Float[Tensor, "d_model"],
    coeff: float = 1.0,
    model_card: Optional["ModelCard"] = None
):
    """
    Create hook that adds direction WITHOUT weighting by expert's routing probability.

    Unlike the weighted version, this simply adds coeff * direction to the MLP output
    broadcast across batch and sequence dimensions.

    Args:
        direction: Steering direction to add [d_model]
        coeff: Coefficient for steering strength (positive or negative)
        model_card: Optional ModelCard for output parsing

    Returns:
        Hook function
    """
    def hook_fn(module, input, output):
        # Parse output using model card if available
        if model_card is not None:
            mlp_output, router_logits = model_card.parse_mlp_output(output)
        else:
            if not isinstance(output, tuple) or len(output) < 2:
                return output
            mlp_output, router_logits = output[0], output[1]

        if router_logits is None:
            return output

        # Add direction directly (broadcast across batch and seq dimensions)
        modified_output = mlp_output + coeff * direction.to(mlp_output.device, mlp_output.dtype)

        # Wrap output
        if model_card is not None:
            return model_card.wrap_mlp_output(modified_output, router_logits)
        else:
            return (modified_output, router_logits)

    return hook_fn


def get_expert_unweighted_intervention_hooks(
    model_base,
    layer_idx: int,
    expert_id: int,
    direction: Float[Tensor, "d_model"],
    coeff: float = 1.0,
    model_card: Optional["ModelCard"] = None
) -> Tuple[list, list]:
    """
    Get hooks for unweighted expert activation addition intervention.

    Same interface as get_expert_weighted_intervention_hooks but uses unweighted hook.
    expert_id is accepted for API compatibility but not used in the hook itself.

    Returns:
        (fwd_pre_hooks, fwd_hooks) tuple
    """
    if model_card is None:
        from model_utils.model_card_factory import create_model_card
        model_card = create_model_card(model_base)

    mlp_module = model_card.get_mlp_module(layer_idx)

    hook_fn = get_expert_unweighted_activation_addition_hook(
        direction=direction,
        coeff=coeff,
        model_card=model_card
    )

    fwd_pre_hooks = []
    fwd_hooks = [(mlp_module, hook_fn)]

    return fwd_pre_hooks, fwd_hooks


# =============================================================================
# Method 2: All-expert weighted steering hooks
# =============================================================================

def get_all_expert_weighted_activation_addition_hook(
    directions_for_layer: Float[Tensor, "n_experts d_model"],
    coeff: float = 1.0,
    model_card: Optional["ModelCard"] = None
):
    """
    Create hook that applies weighted sum of ALL expert steering vectors.

    For each token, reads router probabilities and computes:
        modification = router_probs @ directions_for_layer
    Then adds coeff * modification to MLP output.

    Args:
        directions_for_layer: Steering directions for all experts [n_experts, d_model]
        coeff: Coefficient for steering strength
        model_card: Optional ModelCard for output parsing

    Returns:
        Hook function
    """
    def hook_fn(module, input, output):
        # Parse output
        if model_card is not None:
            mlp_output, router_logits = model_card.parse_mlp_output(output)
        else:
            if not isinstance(output, tuple) or len(output) < 2:
                return output
            mlp_output, router_logits = output[0], output[1]

        if router_logits is None:
            return output

        batch_size, seq_len, d_model = mlp_output.shape

        # Get routing probabilities: [batch*seq, n_experts]
        router_probs = torch.nn.functional.softmax(router_logits, dim=-1)

        # Compute weighted sum: [batch*seq, n_experts] @ [n_experts, d_model] -> [batch*seq, d_model]
        dirs = directions_for_layer.to(mlp_output.device, mlp_output.dtype)
        modification = router_probs @ dirs  # [batch*seq, d_model]

        # Reshape to [batch, seq, d_model]
        modification = modification.view(batch_size, seq_len, d_model)

        # Add to MLP output
        modified_output = mlp_output + coeff * modification

        # Wrap output
        if model_card is not None:
            return model_card.wrap_mlp_output(modified_output, router_logits)
        else:
            return (modified_output, router_logits)

    return hook_fn


def get_all_expert_weighted_intervention_hooks(
    model_base,
    layer_idx: int,
    directions_for_layer: Float[Tensor, "n_experts d_model"],
    coeff: float = 1.0,
    model_card: Optional["ModelCard"] = None
) -> Tuple[list, list]:
    """
    Get hooks for all-expert weighted activation addition intervention.

    Args:
        model_base: The model
        layer_idx: Which layer to intervene at
        directions_for_layer: Steering directions for all experts at this layer [n_experts, d_model]
        coeff: Coefficient (positive to enhance, negative to suppress)
        model_card: Optional ModelCard (created if not provided)

    Returns:
        (fwd_pre_hooks, fwd_hooks) tuple
    """
    if model_card is None:
        from model_utils.model_card_factory import create_model_card
        model_card = create_model_card(model_base)

    mlp_module = model_card.get_mlp_module(layer_idx)

    hook_fn = get_all_expert_weighted_activation_addition_hook(
        directions_for_layer=directions_for_layer,
        coeff=coeff,
        model_card=model_card
    )

    fwd_pre_hooks = []
    fwd_hooks = [(mlp_module, hook_fn)]

    return fwd_pre_hooks, fwd_hooks
