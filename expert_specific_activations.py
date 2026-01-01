"""
Extract expert-specific activations by forcing individual experts.

This module implements activation extraction where each expert is forced
via router bias modification, allowing us to compute mean differences
per expert rather than per layer.
"""

import torch
import os
from typing import List, Tuple
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase


def get_mlp_output_hook(
    layer: int,
    cache: Float[Tensor, "layer pos n d_model"],
    batch_slice: slice,
    positions: List[int]
):
    """
    Hook to capture MLP OUTPUT activations (after expert processing).

    This is different from Arditi's approach which captures block inputs.
    Here we want the MLP output to get expert-specific activations.
    """
    def hook_fn(module, input, output):
        # MLP output is tuple: (hidden_states, router_logits)
        # We want hidden_states (expert outputs)
        if isinstance(output, tuple):
            mlp_output = output[0]  # Shape: (B, S, d)
        else:
            mlp_output = output

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
    force_strength: float = 100.0
) -> Tensor:
    """
    Force a specific expert by adding large bias to its router logits.

    Args:
        model_base: The model
        layer_idx: Which layer to modify
        expert_id: Which expert to force
        force_strength: How much to boost the expert's bias (default: 100.0)

    Returns:
        Original bias tensor (for restoration)
    """
    # Navigate to the router
    # model.model.layers[layer_idx].mlp.router
    router = model_base.model.model.layers[layer_idx].mlp.router

    # Save original bias
    original_bias = router.bias.data.clone()

    # Create modified bias: set all to very negative, then boost target expert
    modified_bias = torch.full_like(original_bias, -force_strength)
    modified_bias[expert_id] = force_strength

    # Apply modification
    router.bias.data = modified_bias

    return original_bias


def restore_router_bias(
    model_base: ModelBase,
    layer_idx: int,
    original_bias: Tensor
):
    """Restore original router bias."""
    router = model_base.model.model.layers[layer_idx].mlp.router
    router.bias.data = original_bias


def get_expert_activations(
    model_base: ModelBase,
    instructions: List[str],
    layer_idx: int,
    expert_id: int,
    *,
    batch_size: int = 32,
    dtype: torch.dtype = torch.float32
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

    Returns:
        Activations with shape [n_positions, n_instructions, d_model]
    """
    model = model_base.model
    tokenizer = model_base.tokenizer
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
    original_bias = force_expert_via_bias(model_base, layer_idx, expert_id)

    try:
        for start in tqdm(range(0, n_instructions, batch_size),
                         desc=f"  Layer {layer_idx} Expert {expert_id}",
                         leave=False):
            end = min(start + batch_size, n_instructions)
            batch_slice = slice(start, end)

            inputs = tokenize_instructions_fn(instructions=instructions[start:end])

            # Hook only the specific MLP layer
            mlp_module = model_base.model.model.layers[layer_idx].mlp
            fwd_hooks = [(
                mlp_module,
                get_mlp_output_hook(
                    layer=0,  # We only have one layer in cache
                    cache=cache,
                    batch_slice=batch_slice,
                    positions=positions
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
        restore_router_bias(model_base, layer_idx, original_bias)
        print(f"  Restored router bias for Layer {layer_idx}")

    # Return shape: [n_positions, n_instructions, d_model]
    return cache[0]


def get_mean_expert_activations(
    model_base: ModelBase,
    instructions: List[str],
    layer_idx: int,
    expert_id: int,
    *,
    batch_size: int = 32
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
        dtype=torch.float64  # High precision for mean computation
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
    batch_size: int = 32
) -> Float[Tensor, "pos d_model"]:
    """
    Compute mean difference for a specific expert.

    Returns:
        Mean difference with shape [n_positions, d_model]
    """
    print(f"\nComputing mean activations for Layer {layer_idx}, Expert {expert_id}...")

    print("  Processing harmful instructions...")
    mean_harmful = get_mean_expert_activations(
        model_base, harmful_instructions, layer_idx, expert_id, batch_size=batch_size
    )

    print("  Processing harmless instructions...")
    mean_harmless = get_mean_expert_activations(
        model_base, harmless_instructions, layer_idx, expert_id, batch_size=batch_size
    )

    mean_diff = mean_harmful - mean_harmless

    print(f"  Mean diff shape: {mean_diff.shape}")
    print(f"  Mean diff magnitude: {mean_diff.norm(dim=-1).mean().item():.4f}")

    return mean_diff


if __name__ == "__main__":
    # Quick test
    print("Testing expert-specific activation extraction...")
    print("This module should be used via run_pipeline_expert_specific.py")
