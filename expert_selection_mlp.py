"""
Selection function for expert-specific directions.

This is adapted from pipeline/submodules/select_direction.py but modified to:
1. Test directions by adding to MLP OUTPUT (not block input)
2. Work with expert-specific (layer, expert) tuples

The key difference: Arditi extracts from and tests at block input,
but we extract from MLP output (after expert processing), so we must
test at the same location.
"""

import json
import torch
import functools
import math
import matplotlib.pyplot as plt
import os

from typing import List, Optional
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.hook_utils import add_hooks
from pipeline.submodules.select_direction import (
    refusal_score,
    get_refusal_scores,
    get_last_position_logits,
    plot_refusal_scores,
    filter_fn,
    kl_div_fn
)
from expert_intervention import (
    get_expert_weighted_activation_addition_hook,
    get_expert_weighted_ablation_hook
)


def get_mlp_activation_addition_hook(
    direction: Float[Tensor, "d_model"],
    coeff: float = 1.0
):
    """
    Hook that adds a direction to MLP output (not block input).

    This is different from Arditi's get_activation_addition_input_pre_hook
    which adds to block input. We add to MLP output where we extracted.
    """
    def hook_fn(module, input, output):
        # MLP output is tuple: (hidden_states, router_logits)
        if isinstance(output, tuple):
            mlp_output, router_logits = output[0], output[1]
        else:
            mlp_output = output
            router_logits = None

        # Add direction to all positions
        # direction: [d_model]
        # mlp_output: [batch, seq, d_model]
        modified_output = mlp_output + coeff * direction.to(mlp_output.device, mlp_output.dtype)

        if router_logits is not None:
            return (modified_output, router_logits)
        else:
            return modified_output

    return hook_fn


def get_mlp_subspace_ablation_hook(
    direction: Float[Tensor, "d_model 1"],
    mu_b: Float[Tensor, "d_model"],
    tau: float = 1.0
):
    """
    Hook that ablates a direction from MLP output.

    Mirrors get_subspace_ablation_output_hook but for MLP output.
    """
    def hook_fn(module, input, output):
        # MLP output is tuple: (hidden_states, router_logits)
        if isinstance(output, tuple):
            mlp_output, router_logits = output[0], output[1]
        else:
            mlp_output = output
            router_logits = None

        # Ablate direction from output
        # direction: [d_model, 1]
        # mlp_output: [batch, seq, d_model]
        # mu_b: [d_model]

        # Center around mu_b
        centered = mlp_output - mu_b.to(mlp_output.device, mlp_output.dtype)

        # Project onto direction and subtract
        # centered @ direction = [batch, seq, 1]
        projection = torch.matmul(centered, direction.to(mlp_output.device, mlp_output.dtype))

        # direction.T = [1, d_model]
        # projection @ direction.T = [batch, seq, d_model]
        ablated = centered - tau * projection * direction.squeeze(-1).to(mlp_output.device, mlp_output.dtype)

        # Add back mu_b
        modified_output = ablated + mu_b.to(mlp_output.device, mlp_output.dtype)

        if router_logits is not None:
            return (modified_output, router_logits)
        else:
            return modified_output

    return hook_fn


def select_expert_direction(
    model_base: ModelBase,
    harmful_instructions,
    harmless_instructions,
    candidate_directions: Float[Tensor, 'n_pos n_candidates d_model'],
    candidate_mapping: dict,  # Maps candidate_idx -> (layer, expert)
    artifact_dir,
    coeff,
    mu_b,
    tau,
    kl_threshold=0.1,
    induce_refusal_threshold=0.0,
    prune_layer_percentage=0.2,
    batch_size=32
):
    """
    Select best expert-specific direction by testing at MLP output level.

    Args:
        candidate_directions: [n_pos, n_candidates, d_model]
        candidate_mapping: {candidate_idx: (layer, expert)}

    Returns:
        pos, candidate_idx, direction
    """

    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    n_pos, n_candidates, d_model = candidate_directions.shape
    n_layer = model_base.model.config.num_hidden_layers

    print("harmful_length:", len(harmful_instructions))
    print("harmless_length:", len(harmless_instructions))

    # Get baseline refusal scores
    baseline_refusal_scores_harmful = get_refusal_scores(
        model_base.model, harmful_instructions,
        model_base.tokenize_instructions_fn, model_base.refusal_toks,
        fwd_hooks=[], batch_size=batch_size,
        tokenizer=model_base.tokenizer,
        refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
    )

    baseline_refusal_scores_harmless = get_refusal_scores(
        model_base.model, harmless_instructions,
        model_base.tokenize_instructions_fn, model_base.refusal_toks,
        fwd_hooks=[], batch_size=batch_size,
        tokenizer=model_base.tokenizer,
        refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
    )

    # Storage for scores
    ablation_kl_div_scores = torch.zeros((n_pos, n_candidates), device=model_base.model.device, dtype=torch.float64)
    ablation_refusal_scores = torch.zeros((n_pos, n_candidates), device=model_base.model.device, dtype=torch.float64)
    steering_refusal_scores = torch.zeros((n_pos, n_candidates), device=model_base.model.device, dtype=torch.float64)

    # Get baseline logits for KL computation
    print("Collecting baseline_harmless_logits")
    baseline_harmless_logits = get_last_position_logits(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=harmless_instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size
    )

    # Test each candidate direction
    for source_pos in range(-n_pos, 0):
        for candidate_idx in tqdm(range(n_candidates), desc=f"Computing KL for position {source_pos}"):
            layer, expert = candidate_mapping[candidate_idx]

            ablation_dir = candidate_directions[source_pos, candidate_idx].unsqueeze(-1)

            # Get MLP module for this layer
            mlp_module = model_base.model.model.layers[layer].mlp

            # Create weighted ablation hooks for MLP output
            ablation_hook = get_expert_weighted_ablation_hook(
                direction=ablation_dir,
                expert_id=expert,
                mu_b=mu_b,
                tau=tau
            )
            fwd_hooks = [(mlp_module, ablation_hook)]

            intervention_logits = get_last_position_logits(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                instructions=harmless_instructions,
                tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                fwd_pre_hooks=[],
                fwd_hooks=fwd_hooks,
                batch_size=batch_size
            )

            ablation_kl_div_scores[source_pos, candidate_idx] = kl_div_fn(
                baseline_harmless_logits, intervention_logits, mask=None
            ).mean(dim=0).item()

    # Compute ablation refusal scores (on harmful)
    for source_pos in range(-n_pos, 0):
        for candidate_idx in tqdm(range(n_candidates), desc=f"Computing refusal ablation for position {source_pos}"):
            layer, expert = candidate_mapping[candidate_idx]

            ablation_dir = candidate_directions[source_pos, candidate_idx].unsqueeze(-1)
            mlp_module = model_base.model.model.layers[layer].mlp

            # Use weighted ablation
            ablation_hook = get_expert_weighted_ablation_hook(
                direction=ablation_dir,
                expert_id=expert,
                mu_b=mu_b,
                tau=tau
            )
            fwd_hooks = [(mlp_module, ablation_hook)]

            refusal_scores = get_refusal_scores(
                model_base.model, harmful_instructions,
                model_base.tokenize_instructions_fn, model_base.refusal_toks,
                fwd_pre_hooks=[], fwd_hooks=fwd_hooks,
                batch_size=batch_size, tokenizer=model_base.tokenizer,
                refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
            )
            ablation_refusal_scores[source_pos, candidate_idx] = refusal_scores.mean().item()

    # Compute steering refusal scores (on harmless)
    for source_pos in range(-n_pos, 0):
        for candidate_idx in tqdm(range(n_candidates), desc=f"Computing refusal addition for position {source_pos}"):
            layer, expert = candidate_mapping[candidate_idx]

            refusal_vector = candidate_directions[source_pos, candidate_idx]
            mlp_module = model_base.model.model.layers[layer].mlp

            # Add direction to MLP output, weighted by expert's routing probability
            addition_hook = get_expert_weighted_activation_addition_hook(
                direction=refusal_vector,
                expert_id=expert,
                coeff=coeff
            )
            fwd_hooks = [(mlp_module, addition_hook)]

            refusal_scores = get_refusal_scores(
                model_base.model, harmless_instructions,
                model_base.tokenize_instructions_fn, model_base.refusal_toks,
                fwd_pre_hooks=[], fwd_hooks=fwd_hooks,
                batch_size=batch_size, tokenizer=model_base.tokenizer,
                refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
            )
            steering_refusal_scores[source_pos, candidate_idx] = refusal_scores.mean().item()

    # Save plots
    plot_refusal_scores(
        refusal_scores=ablation_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmful.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='Ablating expert direction on harmful instructions',
        artifact_dir=artifact_dir,
        artifact_name='expert_ablation_scores'
    )

    plot_refusal_scores(
        refusal_scores=steering_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmless.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='Adding expert direction on harmless instructions',
        artifact_dir=artifact_dir,
        artifact_name='expert_actadd_scores'
    )

    plot_refusal_scores(
        refusal_scores=ablation_kl_div_scores,
        baseline_refusal_score=0.0,
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='KL Divergence when ablating expert direction on harmless',
        artifact_dir=artifact_dir,
        artifact_name='expert_kl_div_scores'
    )

    # Filter and select
    filtered_scores = []
    json_output_all_scores = []
    json_output_filtered_scores = []

    for source_pos in range(-n_pos, 0):
        for candidate_idx in range(n_candidates):
            layer, expert = candidate_mapping[candidate_idx]

            refusal_score = ablation_refusal_scores[source_pos, candidate_idx].item()
            steering_score = steering_refusal_scores[source_pos, candidate_idx].item()
            kl_div_score = ablation_kl_div_scores[source_pos, candidate_idx].item()

            json_output_all_scores.append({
                'position': source_pos,
                'candidate_idx': candidate_idx,
                'layer': layer,
                'expert': expert,
                'refusal_score': refusal_score,
                'steering_score': steering_score,
                'kl_div_score': kl_div_score
            })

            # Sort by negative refusal score (lower is better)
            sorting_score = -refusal_score

            # Filter using candidate_idx as "layer" for filter_fn
            discard_direction = filter_fn(
                refusal_score=refusal_score,
                steering_score=steering_score,
                kl_div_score=kl_div_score,
                layer=candidate_idx,  # Use candidate_idx instead of layer
                n_layer=n_candidates,
                kl_threshold=kl_threshold,
                induce_refusal_threshold=induce_refusal_threshold,
                prune_layer_percentage=prune_layer_percentage
            )

            if discard_direction:
                continue

            filtered_scores.append((sorting_score, source_pos, candidate_idx))

            json_output_filtered_scores.append({
                'position': source_pos,
                'candidate_idx': candidate_idx,
                'layer': layer,
                'expert': expert,
                'refusal_score': refusal_score,
                'steering_score': steering_score,
                'kl_div_score': kl_div_score
            })

    # Save JSON outputs
    with open(f"{artifact_dir}/expert_direction_evaluations.json", 'w') as f:
        json.dump(json_output_all_scores, f, indent=4)

    json_output_filtered_scores = sorted(
        json_output_filtered_scores,
        key=lambda x: (x["refusal_score"], -x["steering_score"], x["kl_div_score"])
    )

    with open(f"{artifact_dir}/expert_direction_evaluations_filtered.json", 'w') as f:
        json.dump(json_output_filtered_scores, f, indent=4)

    assert len(json_output_filtered_scores) > 0, "All scores have been filtered out!"

    # Select best direction
    best = json_output_filtered_scores[0]
    pos = best["position"]
    candidate_idx = best["candidate_idx"]
    layer = best["layer"]
    expert = best["expert"]

    print(f"\n✓ Selected expert direction:")
    print(f"  Position: {pos}")
    print(f"  Candidate: {candidate_idx}")
    print(f"  Layer: {layer}")
    print(f"  Expert: {expert}")
    print(f"  Refusal score: {best['refusal_score']:.4f} (baseline: {baseline_refusal_scores_harmful.mean().item():.4f})")
    print(f"  Steering score: {best['steering_score']:.4f} (baseline: {baseline_refusal_scores_harmless.mean().item():.4f})")
    print(f"  KL Divergence: {best['kl_div_score']:.4f}")

    return pos, candidate_idx, candidate_directions[pos, candidate_idx]
