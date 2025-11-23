import json
import torch
import os

from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.model_utils.model_base import ModelBase
from pipeline.submodules.ica_functions import ica_whitened, get_lp_slices, align_sign_by_means
from pipeline.submodules.select_direction import (
    get_refusal_scores,
    get_last_position_logits,
    kl_div_fn,
    filter_fn,
    plot_refusal_scores
)
from pipeline.utils.hook_utils import (
    get_subspace_ablation_input_pre_hook,
    get_subspace_ablation_output_hook,
    get_activation_addition_subspace_input_pre_hook
)


def select_ica_with_arditi(
    model_base: ModelBase,
    harmful_val: List[str],
    harmless_val: List[str],
    harmful_acts: Float[Tensor, "n_layer n_pos n_harmful d_model"],
    harmless_acts: Float[Tensor, "n_layer n_pos n_harmless d_model"],
    artifact_dir: str,
    topk: int = 1,
    n_components: int = None,
    keep_var: float = 0.999,
    eig_floor_frac: float = 1e-3,
    align: bool = False,
    coeff: float = 1.0,
    mu_b: float = 0.0,
    tau: float = 1.0,
    kl_threshold: float = 0.1,
    induce_refusal_threshold: float = 0.0,
    prune_layer_percentage: float = 0.2,
    batch_size: int = 32
):
    """
    Compute ICA at each layer/position, then select the best using Arditi's empirical criteria.

    Parallel to select_cpca_with_arditi, but uses ICA instead of cPCA.

    Args:
        model_base: The model wrapper
        harmful_val: List of harmful validation instructions
        harmless_val: List of harmless validation instructions
        harmful_acts: Harmful activations [n_layer, n_pos, n_harmful, d_model]
        harmless_acts: Harmless activations [n_layer, n_pos, n_harmless, d_model]
        artifact_dir: Directory to save artifacts
        topk: Number of ICA components to use for intervention
        n_components: Number of ICA components to extract (None = auto)
        keep_var: Variance to keep in background PCA
        eig_floor_frac: Eigenvalue floor fraction
        align: Whether to align component signs
        coeff, mu_b, tau: Intervention parameters
        kl_threshold, induce_refusal_threshold, prune_layer_percentage: Filtering parameters
        batch_size: Batch size for evaluation

    Returns:
        Tuple of (layer, position, components, mu_b)
    """
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    n_layer, n_pos = harmful_acts.shape[:2]

    # Auto-determine n_components if not specified
    if n_components is None:
        n_components = min(topk * 5, 50)

    # Compute baseline refusal scores
    baseline_refusal_scores_harmful = get_refusal_scores(
        model_base.model,
        harmful_val,
        model_base.tokenize_instructions_fn,
        model_base.refusal_toks,
        batch_size=batch_size,
        tokenizer=model_base.tokenizer
    )

    baseline_refusal_scores_harmless = get_refusal_scores(
        model_base.model,
        harmless_val,
        model_base.tokenize_instructions_fn,
        model_base.refusal_toks,
        batch_size=batch_size,
        tokenizer=model_base.tokenizer
    )

    # Get baseline harmless logits for KL computation
    baseline_harmless_logits = get_last_position_logits(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=harmless_val,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        batch_size=batch_size
    )

    # Storage for scores
    ablation_kl_div_scores = torch.zeros((n_pos, n_layer), dtype=torch.float64)
    ablation_refusal_scores = torch.zeros((n_pos, n_layer), dtype=torch.float64)
    steering_refusal_scores = torch.zeros((n_pos, n_layer), dtype=torch.float64)

    # Storage for ICA results
    ica_results = {}  # key: (pos, layer), value: ica result dict

    print("Computing ICA and evaluating with Arditi criteria...")

    # Compute ICA and evaluate at each position/layer
    for source_pos in range(-n_pos, 0):
        print(f"\nProcessing position {source_pos}...")

        for source_layer in tqdm(range(n_layer), desc=f"Position {source_pos}"):
            # Get activations at this layer/position
            Xt, Xb = get_lp_slices(harmful_acts, harmless_acts, source_layer, source_pos)

            # Compute ICA
            res = ica_whitened(
                Xt, Xb,
                n_components=n_components,
                keep_var=keep_var,
                eig_floor_frac=eig_floor_frac
            )

            # Align signs if requested
            if align:
                res["components"], _ = align_sign_by_means(res["components"], Xt, Xb, res["mu_b"])
                res["components_norm"], _ = align_sign_by_means(res["components_norm"], Xt, Xb, res["mu_b"])

            # Store result
            ica_results[(source_pos, source_layer)] = res

            # Extract top-k components as direction(s)
            # Sort by discrimination score and take top-k
            sorted_indices = torch.argsort(res["discrimination_scores"], descending=True)
            top_indices = sorted_indices[:topk]

            direction = res["components_norm"][:, top_indices].to(model_base.model.device, dtype=torch.float32)  # [d_model, topk]
            mu_b_vec = res["mu_b"].to(model_base.model.device, dtype=torch.float32)

            # 1. Evaluate KL divergence with ablation on harmless
            fwd_pre_hooks = [(model_base.model_block_modules[layer],
                             get_subspace_ablation_input_pre_hook(direction, mu_b=mu_b_vec, tau=tau))
                            for layer in range(model_base.model.config.num_hidden_layers)]
            fwd_hooks = [(model_base.model_attn_modules[layer],
                         get_subspace_ablation_output_hook(direction, mu_b=mu_b_vec, tau=tau))
                        for layer in range(model_base.model.config.num_hidden_layers)]
            fwd_hooks += [(model_base.model_mlp_modules[layer],
                          get_subspace_ablation_output_hook(direction, mu_b=mu_b_vec, tau=tau))
                         for layer in range(model_base.model.config.num_hidden_layers)]

            intervention_logits = get_last_position_logits(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                instructions=harmless_val,
                tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
                batch_size=batch_size
            )

            ablation_kl_div_scores[source_pos, source_layer] = kl_div_fn(
                baseline_harmless_logits, intervention_logits, mask=None
            ).mean(dim=0).item()

            # 2. Evaluate refusal score reduction with ablation on harmful
            refusal_scores = get_refusal_scores(
                model_base.model,
                harmful_val,
                model_base.tokenize_instructions_fn,
                model_base.refusal_toks,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
                batch_size=batch_size,
                tokenizer=model_base.tokenizer
            )
            ablation_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()

            # 3. Evaluate steering (inducing refusal on harmless)
            fwd_pre_hooks_steer = [(model_base.model_block_modules[source_layer],
                                   get_activation_addition_subspace_input_pre_hook(direction, coeff=torch.tensor(+coeff)))]
            fwd_hooks_steer = []

            refusal_scores_steer = get_refusal_scores(
                model_base.model,
                harmless_val,
                model_base.tokenize_instructions_fn,
                model_base.refusal_toks,
                fwd_pre_hooks=fwd_pre_hooks_steer,
                fwd_hooks=fwd_hooks_steer,
                batch_size=batch_size,
                tokenizer=model_base.tokenizer
            )
            steering_refusal_scores[source_pos, source_layer] = refusal_scores_steer.mean().item()

    # Plot scores
    plot_refusal_scores(
        refusal_scores=ablation_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmful.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='Ablating ICA direction on harmful instructions',
        artifact_dir=artifact_dir,
        artifact_name='ica_ablation_scores'
    )

    plot_refusal_scores(
        refusal_scores=steering_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmless.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='Adding ICA direction on harmless instructions',
        artifact_dir=artifact_dir,
        artifact_name='ica_actadd_scores'
    )

    plot_refusal_scores(
        refusal_scores=ablation_kl_div_scores,
        baseline_refusal_score=0.0,
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='KL Divergence when ablating ICA direction on harmless instructions',
        artifact_dir=artifact_dir,
        artifact_name='ica_kl_div_scores'
    )

    # Filter and select best
    json_output_all_scores = []
    json_output_filtered_scores = []

    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):
            refusal_score = ablation_refusal_scores[source_pos, source_layer].item()
            steering_score = steering_refusal_scores[source_pos, source_layer].item()
            kl_div_score = ablation_kl_div_scores[source_pos, source_layer].item()

            json_output_all_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'refusal_score': refusal_score,
                'steering_score': steering_score,
                'kl_div_score': kl_div_score
            })

            # Apply Arditi's filtering
            discard = filter_fn(
                refusal_score=refusal_score,
                steering_score=steering_score,
                kl_div_score=kl_div_score,
                layer=source_layer,
                n_layer=n_layer,
                kl_threshold=kl_threshold,
                induce_refusal_threshold=induce_refusal_threshold,
                prune_layer_percentage=prune_layer_percentage
            )

            if not discard:
                json_output_filtered_scores.append({
                    'position': source_pos,
                    'layer': source_layer,
                    'refusal_score': refusal_score,
                    'steering_score': steering_score,
                    'kl_div_score': kl_div_score
                })

    # Save scores
    with open(f"{artifact_dir}/ica_direction_evaluations.json", 'w') as f:
        json.dump(json_output_all_scores, f, indent=4)

    # Sort by Arditi's criteria: lowest refusal score, highest steering score, lowest KL
    json_output_filtered_scores = sorted(
        json_output_filtered_scores,
        key=lambda x: (x["refusal_score"], -x["steering_score"], x["kl_div_score"])
    )

    with open(f"{artifact_dir}/ica_direction_evaluations_filtered.json", 'w') as f:
        json.dump(json_output_filtered_scores, f, indent=4)

    assert len(json_output_filtered_scores) > 0, "All directions have been filtered out!"

    # Select best
    best = json_output_filtered_scores[0]
    pos, layer = best["position"], best["layer"]

    print(f"\nSelected ICA direction: position={pos}, layer={layer}")
    print(f"Refusal score: {ablation_refusal_scores[pos, layer]:.4f} (baseline: {baseline_refusal_scores_harmful.mean().item():.4f})")
    print(f"Steering score: {steering_refusal_scores[pos, layer]:.4f} (baseline: {baseline_refusal_scores_harmless.mean().item():.4f})")
    print(f"KL Divergence: {ablation_kl_div_scores[pos, layer]:.4f}")

    # Return selected ICA result
    selected_res = ica_results[(pos, layer)]

    # Get top-k components by discrimination score
    sorted_indices = torch.argsort(selected_res["discrimination_scores"], descending=True)
    top_indices = sorted_indices[:topk]

    components = selected_res["components_norm"][:, top_indices]
    mu_b_selected = selected_res["mu_b"]

    return layer, pos, components, mu_b_selected
