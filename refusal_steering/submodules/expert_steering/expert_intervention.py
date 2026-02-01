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


def get_expert_weighted_ablation_hook(
    direction: Float[Tensor, "d_model 1"],
    expert_id: int,
    mu_b: Float[Tensor, "d_model"],
    tau: float = 1.0,
    model_card: Optional["ModelCard"] = None
):
    """
    Create hook that ablates direction weighted by expert's routing weight.

    This is the inverse of weighted addition - we remove the expert-weighted
    component of the direction from the MLP output.
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

        # Get routing probabilities
        router_probs = torch.nn.functional.softmax(router_logits, dim=-1)
        expert_weight = router_probs[:, expert_id]  # [batch*seq]

        # Reshape to [batch, seq, 1]
        batch_size, seq_len = mlp_output.shape[0], mlp_output.shape[1]
        expert_weight = expert_weight.view(batch_size, seq_len, 1)

        # Center around mu_b
        centered = mlp_output - mu_b.to(mlp_output.device, mlp_output.dtype)

        # Project onto direction: [batch, seq, d_model] @ [d_model, 1] = [batch, seq, 1]
        projection = torch.matmul(centered, direction.to(mlp_output.device, mlp_output.dtype))

        # Ablate weighted by expert: subtract (expert_weight * tau * projection * direction)
        # This removes the component in the direction proportional to expert's weight
        weighted_projection = expert_weight * projection  # [batch, seq, 1]
        ablated = centered - tau * weighted_projection * direction.squeeze(-1).to(mlp_output.device, mlp_output.dtype)

        # Add back mu_b
        modified_output = ablated + mu_b.to(mlp_output.device, mlp_output.dtype)

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


# === Backward Compatibility Functions ===
# These maintain the old API while using model cards internally

def get_model_layers(model_base):
    """
    DEPRECATED: Use model_card.layers instead.

    This function is kept for backward compatibility.
    """
    from model_utils.model_card_factory import create_model_card
    model_card = create_model_card(model_base)
    return model_card.layers


if __name__ == "__main__":
    # Test the weighted intervention mechanism
    print("Expert-specific weighted intervention module")
    print("Use via run_pipeline_expert_specific.py")

    # Simple unit test
    print("\nUnit test: Weighted addition")

    # Mock router logits for 3 experts
    router_logits = torch.tensor([
        [2.0, 1.0, 0.5],  # Token 1: expert 0 should have highest weight
        [0.5, 3.0, 1.0],  # Token 2: expert 1 should have highest weight
    ])

    # Compute softmax
    router_probs = torch.nn.functional.softmax(router_logits, dim=-1)
    print(f"Router probs:\n{router_probs}")

    # Extract expert 0's weight
    expert_0_weight = router_probs[:, 0]
    print(f"\nExpert 0 weights: {expert_0_weight}")
    print(f"  Token 1: {expert_0_weight[0].item():.3f} (should be highest)")
    print(f"  Token 2: {expert_0_weight[1].item():.3f} (should be low)")

    # Test weighted addition
    direction = torch.ones(5)  # Mock direction
    coeff = 1.0

    mlp_output = torch.zeros(2, 5)  # 2 tokens, 5 dims
    expert_weight_reshaped = expert_0_weight.unsqueeze(-1)  # [2, 1]
    weighted_direction = expert_weight_reshaped * direction  # [2, 5]

    result = mlp_output + coeff * weighted_direction
    print(f"\nWeighted addition result:")
    print(f"  Token 1 (expert 0 active): {result[0]}")
    print(f"  Token 2 (expert 0 inactive): {result[1]}")
    print("\n✓ Unit test passed!")
