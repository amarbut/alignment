"""
Extract expert-specific activations by forcing individual experts.

This module implements activation extraction where each expert is forced
via router bias modification, allowing us to compute mean differences
per expert rather than per layer.

Refactored to use ModelCard abstraction for model-agnostic MoE access.
"""

import torch
from typing import List, Tuple, Optional, TYPE_CHECKING
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from submodules.arditi.hook_utils import add_hooks
from model_utils.model_base import ModelBase

if TYPE_CHECKING:
    from model_utils.model_card import ModelCard


def get_mlp_output_hook(
    layer: int,
    cache: Float[Tensor, "layer pos n d_model"],
    batch_slice: slice,
    positions: List[int],
    model_card: "ModelCard"
):
    """
    Hook to capture MLP OUTPUT activations (after expert processing).

    Uses model_card to parse MLP output correctly for any model architecture.
    """
    def hook_fn(module, input, output):
        # Use model card to parse output (handles tuple vs tensor)
        mlp_output, _ = model_card.parse_mlp_output(output)

        # Select requested positions -> shape (B, P, d)
        x_pos = mlp_output[:, positions, :].to(cache.dtype)

        # Store in cache[layer, :, batch_slice, :]
        cache[layer, :, batch_slice, :] = x_pos.permute(1, 0, 2).contiguous()

        return None  # Don't modify the output

    return hook_fn


def force_expert_via_bias(
    model_base: ModelBase,
    layer_idx: int,
    expert_id: int,
    force_strength: float = 100.0,
    model_card: Optional["ModelCard"] = None
) -> Tuple[Tensor, "ModelCard"]:
    """
    Force a specific expert by adding large bias to its router logits.

    Args:
        model_base: The model
        layer_idx: Which layer to modify
        expert_id: Which expert to force
        force_strength: How much to boost the expert's bias (default: 100.0)
        model_card: Optional ModelCard (created if not provided)

    Returns:
        Tuple of (original_bias, model_card) for restoration
    """
    # Get or create model card
    if model_card is None:
        from model_utils.model_card_factory import create_model_card
        model_card = create_model_card(model_base)

    # Check this is an MoE layer
    if not model_card.is_moe_layer(layer_idx):
        raise ValueError(f"Layer {layer_idx} is not an MoE layer")

    # Get router via model card
    router = model_card.get_router(layer_idx)

    # Get original bias via model card (pass layer_idx for Mixtral hook-based bias)
    original_bias = model_card.get_router_bias(router, layer_idx=layer_idx).clone()

    # Create modified bias: set all to very negative, then boost target expert
    num_experts = model_card.get_num_experts(layer_idx)
    modified_bias = torch.full((num_experts,), -force_strength,
                              device=original_bias.device,
                              dtype=original_bias.dtype)
    modified_bias[expert_id] = force_strength

    # Apply modification via model card (pass layer_idx for Mixtral hook-based bias)
    model_card.set_router_bias(router, modified_bias, layer_idx=layer_idx)

    return original_bias, model_card


def restore_router_bias(
    model_base: ModelBase,
    layer_idx: int,
    original_bias: Tensor,
    model_card: Optional["ModelCard"] = None
):
    """Restore original router bias."""
    # Get or create model card
    if model_card is None:
        from model_utils.model_card_factory import create_model_card
        model_card = create_model_card(model_base)

    # Get router and restore via model card (pass layer_idx for Mixtral hook-based bias)
    router = model_card.get_router(layer_idx)
    model_card.set_router_bias(router, original_bias, layer_idx=layer_idx)


def get_expert_activations(
    model_base: ModelBase,
    instructions: List[str],
    layer_idx: int,
    expert_id: int,
    *,
    batch_size: int = 32,
    dtype: torch.dtype = torch.float32,
    model_card: Optional["ModelCard"] = None
) -> Float[Tensor, "pos n d_model"]:
    """
    Extract MLP output activations when a specific expert is forced.

    Args:
        model_base: The model
        instructions: List of instruction strings
        layer_idx: Which layer contains the expert
        expert_id: Which expert to force
        batch_size: Batch size for processing
        dtype: Data type for cache
        model_card: Optional ModelCard (created if not provided)

    Returns:
        Activations with shape [n_positions, n_instructions, d_model]
    """
    # Get or create model card
    if model_card is None:
        from model_utils.model_card_factory import create_model_card
        model_card = create_model_card(model_base)

    model = model_base.model
    tokenize_instructions_fn = model_base.tokenize_instructions_fn

    positions = list(range(-5, 0))  # Last 5 token positions

    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_instructions = len(instructions)
    d_model = model.config.hidden_size

    # Cache for this single layer
    cache = torch.empty((1, n_positions, n_instructions, d_model),
                        dtype=dtype, device=model.device)

    # Force the expert
    print(f"  Forcing Layer {layer_idx}, Expert {expert_id}...")
    original_bias, model_card = force_expert_via_bias(
        model_base, layer_idx, expert_id, model_card=model_card
    )

    try:
        for start in tqdm(range(0, n_instructions, batch_size),
                         desc=f"  Layer {layer_idx} Expert {expert_id}",
                         leave=False):
            end = min(start + batch_size, n_instructions)
            batch_slice = slice(start, end)

            inputs = tokenize_instructions_fn(instructions=instructions[start:end])

            # Hook the MLP module via model card
            mlp_module = model_card.get_mlp_module(layer_idx)
            fwd_hooks = [(
                mlp_module,
                get_mlp_output_hook(
                    layer=0,  # We only have one layer in cache
                    cache=cache,
                    batch_slice=batch_slice,
                    positions=positions,
                    model_card=model_card
                )
            )]

            with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
                with torch.inference_mode():
                    model(
                        input_ids=inputs.input_ids.to(model.device),
                        attention_mask=inputs.attention_mask.to(model.device),
                    )

    finally:
        # Always restore original bias
        restore_router_bias(model_base, layer_idx, original_bias, model_card=model_card)
        print(f"  Restored router bias for Layer {layer_idx}")

    # Return shape: [n_positions, n_instructions, d_model]
    return cache[0]


def get_mean_expert_activations(
    model_base: ModelBase,
    instructions: List[str],
    layer_idx: int,
    expert_id: int,
    *,
    batch_size: int = 32,
    model_card: Optional["ModelCard"] = None
) -> Float[Tensor, "pos d_model"]:
    """
    Get mean activations for a specific expert.

    Returns:
        Mean activations with shape [n_positions, d_model]
    """
    activations = get_expert_activations(
        model_base,
        instructions,
        layer_idx,
        expert_id,
        batch_size=batch_size,
        dtype=torch.float64,  # High precision for mean computation
        model_card=model_card
    )

    # Average over instructions dimension
    mean_acts = activations.mean(dim=1)  # Shape: [n_positions, d_model]

    return mean_acts


def get_expert_mean_diff(
    model_base: ModelBase,
    harmful_instructions: List[str],
    harmless_instructions: List[str],
    layer_idx: int,
    expert_id: int,
    *,
    batch_size: int = 32,
    model_card: Optional["ModelCard"] = None
) -> Tuple[Float[Tensor, "pos d_model"], Float[Tensor, "pos"]]:
    """
    Compute mean difference for a specific expert.

    Returns:
        Tuple of:
            - Mean difference with shape [n_positions, d_model]
            - Activation RMS per position with shape [n_positions]
              (element-wise RMS of the expert's typical output, averaged over harmful/harmless)
    """
    # Get or create model card
    if model_card is None:
        from model_utils.model_card_factory import create_model_card
        model_card = create_model_card(model_base)

    print(f"\nComputing mean activations for Layer {layer_idx}, Expert {expert_id}...")

    print("  Processing harmful instructions...")
    mean_harmful = get_mean_expert_activations(
        model_base, harmful_instructions, layer_idx, expert_id,
        batch_size=batch_size, model_card=model_card
    )

    print("  Processing harmless instructions...")
    mean_harmless = get_mean_expert_activations(
        model_base, harmless_instructions, layer_idx, expert_id,
        batch_size=batch_size, model_card=model_card
    )

    mean_diff = mean_harmful - mean_harmless

    # Compute per-position activation RMS: element-wise RMS averaged over harmful/harmless
    # Shape: [n_positions]
    activation_rms = ((mean_harmful**2 + mean_harmless**2) / 2).mean(dim=-1).sqrt()

    print(f"  Mean diff shape: {mean_diff.shape}")
    print(f"  Mean diff magnitude: {mean_diff.norm(dim=-1).mean().item():.4f}")
    print(f"  Activation RMS per position: {activation_rms.tolist()}")

    return mean_diff, activation_rms
