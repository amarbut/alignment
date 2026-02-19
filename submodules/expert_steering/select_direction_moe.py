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
import matplotlib.pyplot as plt
import os

from typing import Optional, TYPE_CHECKING
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from model_utils.model_base import ModelBase

if TYPE_CHECKING:
    from model_utils.model_card import ModelCard

from submodules.arditi.select_direction import (
    get_refusal_scores,
    get_last_position_logits,
    plot_refusal_scores,
    filter_fn,
    kl_div_fn,
)
from submodules.expert_steering.expert_intervention import (
    get_expert_weighted_activation_addition_hook,
    get_expert_unweighted_activation_addition_hook,
    get_all_expert_weighted_activation_addition_hook,
)


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
    kl_threshold=None,  # If None, get from model_card
    induce_refusal_threshold=None,  # If None, get from model_card
    prune_layer_percentage=None,  # If None, get from model_card
    batch_size=32,
    top_n=1,
    model_card: Optional["ModelCard"] = None
):
    """
    Select best expert-specific direction(s) by testing at MLP output level.

    Args:
        candidate_directions: [n_pos, n_candidates, d_model]
        candidate_mapping: {candidate_idx: (layer, expert)}
        top_n: Number of top directions to return (default: 1)

    Returns:
        If top_n == 1: (pos, candidate_idx, direction)
        If top_n > 1: list of (pos, candidate_idx, direction) tuples
    """

    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    # Get thresholds from model_card if not provided
    if model_card is not None and hasattr(model_card, 'get_expert_steering_thresholds'):
        thresholds = model_card.get_expert_steering_thresholds()
        if kl_threshold is None:
            kl_threshold = thresholds.get('kl_threshold', 1.0)
        if induce_refusal_threshold is None:
            induce_refusal_threshold = thresholds.get('steering_score_threshold', -5.0)
        if prune_layer_percentage is None:
            prune_layer_percentage = thresholds.get('prune_layer_percentage', 0.0)
        print(f"Using model-specific thresholds: kl={kl_threshold}, steering={induce_refusal_threshold}, prune={prune_layer_percentage}")
    else:
        # Fallback defaults (OSS-like)
        if kl_threshold is None:
            kl_threshold = 1.0
        if induce_refusal_threshold is None:
            induce_refusal_threshold = -5.0
        if prune_layer_percentage is None:
            prune_layer_percentage = 0.0
        print(f"Using default thresholds: kl={kl_threshold}, steering={induce_refusal_threshold}, prune={prune_layer_percentage}")

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
    # Use NEGATIVE actAdd instead of ablation (which breaks the model)
    # Logic: if we induce refusal (negative coeff), does it mess up logits on harmless?
    for source_pos in range(-n_pos, 0):
        for candidate_idx in tqdm(range(n_candidates), desc=f"Computing KL for position {source_pos}"):
            layer, expert = candidate_mapping[candidate_idx]

            direction_vec = candidate_directions[source_pos, candidate_idx]

            # Get MLP module for this layer
            mlp_module = model_card.get_mlp_module(layer)

            # Use negative actAdd (induce refusal) instead of ablation
            # This should increase refusal on harmless without breaking the model
            negative_add_hook = get_expert_weighted_activation_addition_hook(
                direction=direction_vec,
                expert_id=expert,
                coeff=-coeff,  # Negative to induce refusal
                model_card=model_card
            )
            fwd_hooks = [(mlp_module, negative_add_hook)]

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

    # Compute refusal reduction scores (on harmful)
    # Use NEGATIVE actAdd instead of ablation
    # Logic: negative coeff should reduce refusal on harmful (make model comply)
    for source_pos in range(-n_pos, 0):
        for candidate_idx in tqdm(range(n_candidates), desc=f"Computing refusal reduction for position {source_pos}"):
            layer, expert = candidate_mapping[candidate_idx]

            direction_vec = candidate_directions[source_pos, candidate_idx]
            mlp_module = model_card.get_mlp_module(layer)

            # Use negative actAdd (reduce refusal)
            negative_add_hook = get_expert_weighted_activation_addition_hook(
                direction=direction_vec,
                expert_id=expert,
                coeff=-coeff,  # Negative to reduce refusal
                model_card=model_card
            )
            fwd_hooks = [(mlp_module, negative_add_hook)]

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
            mlp_module = model_card.get_mlp_module(layer)

            # Add direction to MLP output, weighted by expert's routing probability
            addition_hook = get_expert_weighted_activation_addition_hook(
                direction=refusal_vector,
                expert_id=expert,
                coeff=coeff,
                model_card=model_card
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

            refusal_score_val = ablation_refusal_scores[source_pos, candidate_idx].item()
            steering_score = steering_refusal_scores[source_pos, candidate_idx].item()
            kl_div_score = ablation_kl_div_scores[source_pos, candidate_idx].item()

            json_output_all_scores.append({
                'position': source_pos,
                'candidate_idx': candidate_idx,
                'layer': layer,
                'expert': expert,
                'refusal_score': refusal_score_val,
                'steering_score': steering_score,
                'kl_div_score': kl_div_score
            })

            # Sort by negative refusal score (lower is better)
            sorting_score = -refusal_score_val

            # Filter using candidate_idx as "layer" for filter_fn
            discard_direction = filter_fn(
                refusal_score=refusal_score_val,
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
                'refusal_score': refusal_score_val,
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

    # Select top-n directions
    n_to_select = min(top_n, len(json_output_filtered_scores))

    if n_to_select > len(json_output_filtered_scores):
        print(f"\nWarning: Requested top_n={top_n} but only {len(json_output_filtered_scores)} directions passed filtering")

    selected_directions = []

    print(f"\n✓ Selected top {n_to_select} expert direction(s):")
    for i in range(n_to_select):
        entry = json_output_filtered_scores[i]
        pos = entry["position"]
        candidate_idx = entry["candidate_idx"]
        layer = entry["layer"]
        expert = entry["expert"]

        print(f"\n  [{i+1}/{n_to_select}]")
        print(f"    Position: {pos}")
        print(f"    Candidate: {candidate_idx}")
        print(f"    Layer: {layer}")
        print(f"    Expert: {expert}")
        print(f"    Refusal score: {entry['refusal_score']:.4f} (baseline: {baseline_refusal_scores_harmful.mean().item():.4f})")
        print(f"    Steering score: {entry['steering_score']:.4f} (baseline: {baseline_refusal_scores_harmless.mean().item():.4f})")
        print(f"    KL Divergence: {entry['kl_div_score']:.4f}")

        selected_directions.append((pos, candidate_idx, candidate_directions[pos, candidate_idx]))

    # Return single tuple for backward compatibility when top_n=1
    if top_n == 1:
        return selected_directions[0]
    else:
        return selected_directions


def select_expert_direction_unweighted(
    model_base: ModelBase,
    harmful_instructions,
    harmless_instructions,
    candidate_directions: Float[Tensor, 'n_pos n_candidates d_model'],
    candidate_mapping: dict,  # Maps candidate_idx -> (layer, expert)
    artifact_dir,
    coeff,
    mu_b,
    tau,
    kl_threshold=None,
    induce_refusal_threshold=None,
    prune_layer_percentage=None,
    batch_size=32,
    top_n=1,
    model_card: Optional["ModelCard"] = None
):
    """
    Select best expert-specific direction(s) using UNWEIGHTED hooks.

    Same logic as select_expert_direction but uses unweighted activation addition
    (no multiplication by expert routing probability).

    Returns:
        If top_n == 1: (pos, candidate_idx, direction)
        If top_n > 1: list of (pos, candidate_idx, direction) tuples
    """

    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    # Get thresholds from model_card if not provided
    if model_card is not None and hasattr(model_card, 'get_expert_steering_thresholds'):
        thresholds = model_card.get_expert_steering_thresholds()
        if kl_threshold is None:
            kl_threshold = thresholds.get('kl_threshold', 1.0)
        if induce_refusal_threshold is None:
            induce_refusal_threshold = thresholds.get('steering_score_threshold', -5.0)
        if prune_layer_percentage is None:
            prune_layer_percentage = thresholds.get('prune_layer_percentage', 0.0)
        print(f"Using model-specific thresholds: kl={kl_threshold}, steering={induce_refusal_threshold}, prune={prune_layer_percentage}")
    else:
        if kl_threshold is None:
            kl_threshold = 1.0
        if induce_refusal_threshold is None:
            induce_refusal_threshold = -5.0
        if prune_layer_percentage is None:
            prune_layer_percentage = 0.0
        print(f"Using default thresholds: kl={kl_threshold}, steering={induce_refusal_threshold}, prune={prune_layer_percentage}")

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

    # Test each candidate direction with UNWEIGHTED hooks
    for source_pos in range(-n_pos, 0):
        for candidate_idx in tqdm(range(n_candidates), desc=f"Computing KL for position {source_pos}"):
            layer, expert = candidate_mapping[candidate_idx]

            direction_vec = candidate_directions[source_pos, candidate_idx]
            mlp_module = model_card.get_mlp_module(layer)

            # Unweighted hook (no expert routing weight)
            negative_add_hook = get_expert_unweighted_activation_addition_hook(
                direction=direction_vec,
                coeff=-coeff,
                model_card=model_card
            )
            fwd_hooks = [(mlp_module, negative_add_hook)]

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

    # Compute refusal reduction scores (on harmful) with unweighted hooks
    for source_pos in range(-n_pos, 0):
        for candidate_idx in tqdm(range(n_candidates), desc=f"Computing refusal reduction for position {source_pos}"):
            layer, expert = candidate_mapping[candidate_idx]

            direction_vec = candidate_directions[source_pos, candidate_idx]
            mlp_module = model_card.get_mlp_module(layer)

            negative_add_hook = get_expert_unweighted_activation_addition_hook(
                direction=direction_vec,
                coeff=-coeff,
                model_card=model_card
            )
            fwd_hooks = [(mlp_module, negative_add_hook)]

            refusal_scores = get_refusal_scores(
                model_base.model, harmful_instructions,
                model_base.tokenize_instructions_fn, model_base.refusal_toks,
                fwd_pre_hooks=[], fwd_hooks=fwd_hooks,
                batch_size=batch_size, tokenizer=model_base.tokenizer,
                refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
            )
            ablation_refusal_scores[source_pos, candidate_idx] = refusal_scores.mean().item()

    # Compute steering refusal scores (on harmless) with unweighted hooks
    for source_pos in range(-n_pos, 0):
        for candidate_idx in tqdm(range(n_candidates), desc=f"Computing refusal addition for position {source_pos}"):
            layer, expert = candidate_mapping[candidate_idx]

            refusal_vector = candidate_directions[source_pos, candidate_idx]
            mlp_module = model_card.get_mlp_module(layer)

            addition_hook = get_expert_unweighted_activation_addition_hook(
                direction=refusal_vector,
                coeff=coeff,
                model_card=model_card
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
        title='Unweighted: ablating expert direction on harmful instructions',
        artifact_dir=artifact_dir,
        artifact_name='expert_ablation_scores_unweighted'
    )

    plot_refusal_scores(
        refusal_scores=steering_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmless.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='Unweighted: adding expert direction on harmless instructions',
        artifact_dir=artifact_dir,
        artifact_name='expert_actadd_scores_unweighted'
    )

    plot_refusal_scores(
        refusal_scores=ablation_kl_div_scores,
        baseline_refusal_score=0.0,
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='Unweighted: KL Divergence on harmless',
        artifact_dir=artifact_dir,
        artifact_name='expert_kl_div_scores_unweighted'
    )

    # Filter and select (same logic as weighted version)
    filtered_scores = []
    json_output_all_scores = []
    json_output_filtered_scores = []

    for source_pos in range(-n_pos, 0):
        for candidate_idx in range(n_candidates):
            layer, expert = candidate_mapping[candidate_idx]

            refusal_score_val = ablation_refusal_scores[source_pos, candidate_idx].item()
            steering_score = steering_refusal_scores[source_pos, candidate_idx].item()
            kl_div_score = ablation_kl_div_scores[source_pos, candidate_idx].item()

            json_output_all_scores.append({
                'position': source_pos,
                'candidate_idx': candidate_idx,
                'layer': layer,
                'expert': expert,
                'refusal_score': refusal_score_val,
                'steering_score': steering_score,
                'kl_div_score': kl_div_score
            })

            sorting_score = -refusal_score_val

            discard_direction = filter_fn(
                refusal_score=refusal_score_val,
                steering_score=steering_score,
                kl_div_score=kl_div_score,
                layer=candidate_idx,
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
                'refusal_score': refusal_score_val,
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

    n_to_select = min(top_n, len(json_output_filtered_scores))

    selected_directions = []

    print(f"\n✓ Selected top {n_to_select} unweighted expert direction(s):")
    for i in range(n_to_select):
        entry = json_output_filtered_scores[i]
        pos = entry["position"]
        candidate_idx = entry["candidate_idx"]
        layer = entry["layer"]
        expert = entry["expert"]

        print(f"\n  [{i+1}/{n_to_select}]")
        print(f"    Position: {pos}")
        print(f"    Candidate: {candidate_idx}")
        print(f"    Layer: {layer}")
        print(f"    Expert: {expert}")
        print(f"    Refusal score: {entry['refusal_score']:.4f} (baseline: {baseline_refusal_scores_harmful.mean().item():.4f})")
        print(f"    Steering score: {entry['steering_score']:.4f} (baseline: {baseline_refusal_scores_harmless.mean().item():.4f})")
        print(f"    KL Divergence: {entry['kl_div_score']:.4f}")

        selected_directions.append((pos, candidate_idx, candidate_directions[pos, candidate_idx]))

    if top_n == 1:
        return selected_directions[0]
    else:
        return selected_directions


def select_layer_direction(
    model_base: ModelBase,
    harmful_instructions,
    harmless_instructions,
    all_layer_directions: dict,  # {layer_idx: tensor[n_experts, n_pos, d_model]}
    artifact_dir,
    coeff,
    kl_threshold=None,
    induce_refusal_threshold=None,
    prune_layer_percentage=None,
    batch_size=32,
    model_card: Optional["ModelCard"] = None
):
    """
    Select best (layer, position) pair for all-expert weighted steering.

    Searches over (layer, position) candidates. For each candidate, creates
    a hook using all expert directions at that layer/position.

    Args:
        all_layer_directions: {layer_idx: tensor[n_experts, n_pos, d_model]}

    Returns:
        (best_layer, best_pos, directions_for_layer) where directions_for_layer
        is [n_experts, d_model] at the selected position
    """

    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    # Get thresholds from model_card if not provided
    if model_card is not None and hasattr(model_card, 'get_expert_steering_thresholds'):
        thresholds = model_card.get_expert_steering_thresholds()
        if kl_threshold is None:
            kl_threshold = thresholds.get('kl_threshold', 1.0)
        if induce_refusal_threshold is None:
            induce_refusal_threshold = thresholds.get('steering_score_threshold', -5.0)
        if prune_layer_percentage is None:
            prune_layer_percentage = thresholds.get('prune_layer_percentage', 0.0)
        print(f"Using model-specific thresholds: kl={kl_threshold}, steering={induce_refusal_threshold}, prune={prune_layer_percentage}")
    else:
        if kl_threshold is None:
            kl_threshold = 1.0
        if induce_refusal_threshold is None:
            induce_refusal_threshold = -5.0
        if prune_layer_percentage is None:
            prune_layer_percentage = 0.0
        print(f"Using default thresholds: kl={kl_threshold}, steering={induce_refusal_threshold}, prune={prune_layer_percentage}")

    layer_indices = sorted(all_layer_directions.keys())
    n_layers_candidate = len(layer_indices)

    # Get n_pos from first layer's directions
    first_dirs = all_layer_directions[layer_indices[0]]
    n_pos = first_dirs.shape[1]  # [n_experts, n_pos, d_model]

    n_layer_total = model_base.model.config.num_hidden_layers

    print("harmful_length:", len(harmful_instructions))
    print("harmless_length:", len(harmless_instructions))
    print(f"Candidate space: {n_layers_candidate} layers x {n_pos} positions = {n_layers_candidate * n_pos} candidates")

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

    # Storage for scores: indexed by (pos_idx, layer_candidate_idx)
    ablation_kl_div_scores = torch.zeros((n_pos, n_layers_candidate), device=model_base.model.device, dtype=torch.float64)
    ablation_refusal_scores = torch.zeros((n_pos, n_layers_candidate), device=model_base.model.device, dtype=torch.float64)
    steering_refusal_scores = torch.zeros((n_pos, n_layers_candidate), device=model_base.model.device, dtype=torch.float64)

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

    # Test each (layer, position) candidate
    for source_pos in range(-n_pos, 0):
        for layer_cand_idx, layer_idx in enumerate(tqdm(layer_indices, desc=f"Computing KL for position {source_pos}")):
            dirs = all_layer_directions[layer_idx]  # [n_experts, n_pos, d_model]
            dirs_at_pos = dirs[:, source_pos, :]  # [n_experts, d_model]

            mlp_module = model_card.get_mlp_module(layer_idx)

            hook = get_all_expert_weighted_activation_addition_hook(
                directions_for_layer=dirs_at_pos,
                coeff=-coeff,
                model_card=model_card
            )
            fwd_hooks = [(mlp_module, hook)]

            intervention_logits = get_last_position_logits(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                instructions=harmless_instructions,
                tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                fwd_pre_hooks=[],
                fwd_hooks=fwd_hooks,
                batch_size=batch_size
            )

            ablation_kl_div_scores[source_pos, layer_cand_idx] = kl_div_fn(
                baseline_harmless_logits, intervention_logits, mask=None
            ).mean(dim=0).item()

    # Refusal reduction on harmful
    for source_pos in range(-n_pos, 0):
        for layer_cand_idx, layer_idx in enumerate(tqdm(layer_indices, desc=f"Computing refusal reduction for position {source_pos}")):
            dirs = all_layer_directions[layer_idx]
            dirs_at_pos = dirs[:, source_pos, :]

            mlp_module = model_card.get_mlp_module(layer_idx)

            hook = get_all_expert_weighted_activation_addition_hook(
                directions_for_layer=dirs_at_pos,
                coeff=-coeff,
                model_card=model_card
            )
            fwd_hooks = [(mlp_module, hook)]

            refusal_scores = get_refusal_scores(
                model_base.model, harmful_instructions,
                model_base.tokenize_instructions_fn, model_base.refusal_toks,
                fwd_pre_hooks=[], fwd_hooks=fwd_hooks,
                batch_size=batch_size, tokenizer=model_base.tokenizer,
                refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
            )
            ablation_refusal_scores[source_pos, layer_cand_idx] = refusal_scores.mean().item()

    # Steering on harmless
    for source_pos in range(-n_pos, 0):
        for layer_cand_idx, layer_idx in enumerate(tqdm(layer_indices, desc=f"Computing refusal addition for position {source_pos}")):
            dirs = all_layer_directions[layer_idx]
            dirs_at_pos = dirs[:, source_pos, :]

            mlp_module = model_card.get_mlp_module(layer_idx)

            hook = get_all_expert_weighted_activation_addition_hook(
                directions_for_layer=dirs_at_pos,
                coeff=coeff,
                model_card=model_card
            )
            fwd_hooks = [(mlp_module, hook)]

            refusal_scores = get_refusal_scores(
                model_base.model, harmless_instructions,
                model_base.tokenize_instructions_fn, model_base.refusal_toks,
                fwd_pre_hooks=[], fwd_hooks=fwd_hooks,
                batch_size=batch_size, tokenizer=model_base.tokenizer,
                refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
            )
            steering_refusal_scores[source_pos, layer_cand_idx] = refusal_scores.mean().item()

    # Save plots
    plot_refusal_scores(
        refusal_scores=ablation_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmful.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='All-expert: ablating direction on harmful instructions',
        artifact_dir=artifact_dir,
        artifact_name='allex_ablation_scores'
    )

    plot_refusal_scores(
        refusal_scores=steering_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmless.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='All-expert: adding direction on harmless instructions',
        artifact_dir=artifact_dir,
        artifact_name='allex_actadd_scores'
    )

    plot_refusal_scores(
        refusal_scores=ablation_kl_div_scores,
        baseline_refusal_score=0.0,
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='All-expert: KL Divergence on harmless',
        artifact_dir=artifact_dir,
        artifact_name='allex_kl_div_scores'
    )

    # Filter and select best (layer, position) pair
    json_output_all_scores = []
    json_output_filtered_scores = []

    for source_pos in range(-n_pos, 0):
        for layer_cand_idx, layer_idx in enumerate(layer_indices):
            refusal_score_val = ablation_refusal_scores[source_pos, layer_cand_idx].item()
            steering_score = steering_refusal_scores[source_pos, layer_cand_idx].item()
            kl_div_score = ablation_kl_div_scores[source_pos, layer_cand_idx].item()

            json_output_all_scores.append({
                'position': source_pos,
                'layer': layer_idx,
                'refusal_score': refusal_score_val,
                'steering_score': steering_score,
                'kl_div_score': kl_div_score
            })

            discard_direction = filter_fn(
                refusal_score=refusal_score_val,
                steering_score=steering_score,
                kl_div_score=kl_div_score,
                layer=layer_cand_idx,
                n_layer=n_layers_candidate,
                kl_threshold=kl_threshold,
                induce_refusal_threshold=induce_refusal_threshold,
                prune_layer_percentage=prune_layer_percentage
            )

            if discard_direction:
                continue

            json_output_filtered_scores.append({
                'position': source_pos,
                'layer': layer_idx,
                'refusal_score': refusal_score_val,
                'steering_score': steering_score,
                'kl_div_score': kl_div_score
            })

    # Save JSON outputs
    with open(f"{artifact_dir}/layer_direction_evaluations.json", 'w') as f:
        json.dump(json_output_all_scores, f, indent=4)

    json_output_filtered_scores = sorted(
        json_output_filtered_scores,
        key=lambda x: (x["refusal_score"], -x["steering_score"], x["kl_div_score"])
    )

    with open(f"{artifact_dir}/layer_direction_evaluations_filtered.json", 'w') as f:
        json.dump(json_output_filtered_scores, f, indent=4)

    assert len(json_output_filtered_scores) > 0, "All scores have been filtered out!"

    # Best result
    best = json_output_filtered_scores[0]
    best_layer = best["layer"]
    best_pos = best["position"]

    print(f"\n✓ Best all-expert direction:")
    print(f"  Layer: {best_layer}")
    print(f"  Position: {best_pos}")
    print(f"  Refusal score: {best['refusal_score']:.4f} (baseline: {baseline_refusal_scores_harmful.mean().item():.4f})")
    print(f"  Steering score: {best['steering_score']:.4f} (baseline: {baseline_refusal_scores_harmless.mean().item():.4f})")
    print(f"  KL Divergence: {best['kl_div_score']:.4f}")

    # Return the selected directions for this layer at the best position
    dirs = all_layer_directions[best_layer]  # [n_experts, n_pos, d_model]
    directions_for_layer = dirs[:, best_pos, :]  # [n_experts, d_model]

    return best_layer, best_pos, directions_for_layer
