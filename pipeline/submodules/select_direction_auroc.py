import json
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

from typing import List, Optional
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.model_utils.model_base import ModelBase
from pipeline.submodules.generate_activations import get_activations


def compute_projection_fisher(
    harmful_activations: Float[Tensor, "n_harmful d_model"],
    harmless_activations: Float[Tensor, "n_harmless d_model"],
    direction: Float[Tensor, "d_model"]
) -> float:
    """
    Compute Fisher's Linear Discriminant criterion for a direction.

    Fisher criterion = (μ_harmful - μ_harmless)² / (σ²_harmful + σ²_harmless)

    This measures the ratio of between-class variance to within-class variance,
    analogous to the Rayleigh quotient used in cPCA.

    Args:
        harmful_activations: Activations from harmful instructions
        harmless_activations: Activations from harmless instructions
        direction: The direction vector to project onto

    Returns:
        Fisher score (higher is better at distinguishing harmful from harmless)
    """
    # Convert direction to same dtype as activations
    direction = direction.to(dtype=harmful_activations.dtype)

    # Normalize direction
    direction_norm = direction / (direction.norm() + 1e-8)

    # Project activations onto direction
    harmful_projections = (harmful_activations @ direction_norm).cpu().numpy()
    harmless_projections = (harmless_activations @ direction_norm).cpu().numpy()

    # Compute means
    mu_harmful = harmful_projections.mean()
    mu_harmless = harmless_projections.mean()

    # Compute variances (using ddof=1 for unbiased estimate)
    var_harmful = harmful_projections.var(ddof=1)
    var_harmless = harmless_projections.var(ddof=1)

    # Fisher criterion: between-class variance / within-class variance
    between_var = (mu_harmful - mu_harmless) ** 2
    within_var = var_harmful + var_harmless

    # Avoid division by zero
    if within_var < 1e-10:
        return 0.0

    fisher_score = between_var / within_var

    return float(fisher_score)


def plot_fisher_scores(
    fisher_scores: Float[Tensor, 'n_pos n_layer'],
    token_labels: List[str],
    title: str,
    artifact_dir: str,
    artifact_name: str,
):
    """Plot Fisher scores across positions and layers."""
    n_pos, n_layer = fisher_scores.shape

    fig, ax = plt.subplots(figsize=(9, 5))

    # Add a trace for each position
    for i in range(-n_pos, 0):
        ax.plot(
            list(range(n_layer)),
            fisher_scores[i].cpu().numpy(),
            label=f'{i}'
        )

    ax.set_title(title)
    ax.set_xlabel('Layer source of direction (resid_pre)')
    ax.set_ylabel('Fisher Score')
    ax.legend(title='Position source of direction', loc='best')

    plt.savefig(f"{artifact_dir}/{artifact_name}.png")
    plt.close()


def select_direction_auroc(
    model_base: ModelBase,
    harmful_instructions: List[str],
    harmless_instructions: List[str],
    candidate_directions: Float[Tensor, 'n_pos n_layer d_model'],
    artifact_dir: str,
    batch_size: int = 32,
    prune_layer_percentage: float = 0.2,
    min_auroc_threshold: float = 0.5
):
    """
    Select the best refusal direction based on Fisher's Linear Discriminant criterion.

    Fisher criterion = (μ_harmful - μ_harmless)² / (σ²_harmful + σ²_harmless)

    This measures the ratio of between-class variance to within-class variance,
    analogous to the Rayleigh quotient used in cPCA.

    Args:
        model_base: The model wrapper
        harmful_instructions: List of harmful instruction strings
        harmless_instructions: List of harmless instruction strings
        candidate_directions: Tensor of candidate directions [n_pos, n_layer, d_model]
        artifact_dir: Directory to save artifacts
        batch_size: Batch size for activation collection
        prune_layer_percentage: Percentage of final layers to exclude (default 0.2 = last 20%)
        min_auroc_threshold: Minimum Fisher score threshold (default 0.5, kept for backwards compat)

    Returns:
        Tuple of (position, layer, direction vector)
    """
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    n_pos, n_layer, d_model = candidate_directions.shape

    print(f"Collecting activations for {len(harmful_instructions)} harmful instructions...")
    harmful_activations = get_activations(
        model_base,
        harmful_instructions,
        batch_size=batch_size,
        dtype=torch.float32
    )  # [n_layer, n_pos, n_harmful, d_model]

    print(f"Collecting activations for {len(harmless_instructions)} harmless instructions...")
    harmless_activations = get_activations(
        model_base,
        harmless_instructions,
        batch_size=batch_size,
        dtype=torch.float32
    )  # [n_layer, n_pos, n_harmless, d_model]

    # Compute Fisher scores for each candidate direction
    fisher_scores = torch.zeros((n_pos, n_layer), dtype=torch.float32)

    print("Computing Fisher scores for each candidate direction...")
    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Position {source_pos}"):
            # Get activations at this position and layer
            harmful_acts = harmful_activations[source_layer, source_pos]  # [n_harmful, d_model]
            harmless_acts = harmless_activations[source_layer, source_pos]  # [n_harmless, d_model]

            # Get candidate direction
            direction = candidate_directions[source_pos, source_layer]  # [d_model]

            # Compute Fisher score
            fisher_score = compute_projection_fisher(harmful_acts, harmless_acts, direction)
            fisher_scores[source_pos, source_layer] = fisher_score

    # Plot Fisher scores
    plot_fisher_scores(
        fisher_scores=fisher_scores,
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='Fisher scores for candidate directions',
        artifact_dir=artifact_dir,
        artifact_name='fisher_scores'
    )

    # Filter and select best direction
    candidates = []
    json_output_all_scores = []
    json_output_filtered_scores = []

    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):
            fisher_score = fisher_scores[source_pos, source_layer].item()

            json_output_all_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'fisher_score': fisher_score
            })

            # Filter: exclude last layers and low Fisher scores
            if source_layer >= int(n_layer * (1.0 - prune_layer_percentage)):
                continue
            if fisher_score < min_auroc_threshold:  # reusing threshold param name
                continue
            if np.isnan(fisher_score):
                continue

            candidates.append((fisher_score, source_pos, source_layer))
            json_output_filtered_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'fisher_score': fisher_score
            })

    # Save all scores
    with open(f"{artifact_dir}/direction_evaluations_fisher.json", 'w') as f:
        json.dump(json_output_all_scores, f, indent=4)

    # Sort by Fisher score (descending - higher is better)
    json_output_filtered_scores = sorted(
        json_output_filtered_scores,
        key=lambda x: -x["fisher_score"]
    )

    with open(f"{artifact_dir}/direction_evaluations_fisher_filtered.json", 'w') as f:
        json.dump(json_output_filtered_scores, f, indent=4)

    assert len(json_output_filtered_scores) > 0, "All directions have been filtered out!"

    # Select best direction (highest Fisher score)
    best = json_output_filtered_scores[0]
    pos, layer = best["position"], best["layer"]
    fisher_score = best["fisher_score"]

    print(f"\nSelected direction: position={pos}, layer={layer}")
    print(f"Fisher score: {fisher_score:.4f}")
    print(f"Number of candidates after filtering: {len(json_output_filtered_scores)}")

    return pos, layer, candidate_directions[pos, layer]
