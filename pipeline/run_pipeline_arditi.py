"""
Clean Arditi Refusal Direction Pipeline

This is a simplified version of run_pipeline_subspace.py that:
- Only supports the Arditi mean-difference method
- Only implements ActAdd intervention (no ablation)
- Uses enhanced Config with system prompt support
- Saves metadata.json with all hyperparameters

For the full pipeline with cpca/ica methods and ablation, use run_pipeline_subspace.py
"""

import torch
import random
import json
import os
import argparse
from datetime import datetime

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config, SYSTEM_PROMPTS
from pipeline.model_utils.model_factory_moe import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_subspace_input_pre_hook

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Arditi refusal direction pipeline with ActAdd intervention"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the model'
    )
    parser.add_argument(
        '--system_prompt',
        type=str,
        default=None,
        choices=["none", "llama_2", "lightweight"],
        help='System prompt to use (default: use Config default)'
    )
    parser.add_argument(
        '--coeff',
        type=float,
        default=1.0,
        help='Scaling coefficient for ActAdd intervention (default: 1.0)'
    )
    parser.add_argument(
        '--skip_generate',
        action='store_true',
        help='Skip direction generation, load from cache'
    )
    parser.add_argument(
        '--skip_select',
        action='store_true',
        help='Skip direction selection, load from cache'
    )
    parser.add_argument(
        '--skip_baseline',
        action='store_true',
        help='Skip baseline evaluation'
    )
    parser.add_argument(
        '--n_train',
        type=int,
        default=None,
        help='Number of training samples (default: use Config default)'
    )
    parser.add_argument(
        '--n_val',
        type=int,
        default=None,
        help='Number of validation samples (default: use Config default)'
    )
    parser.add_argument(
        '--n_test',
        type=int,
        default=None,
        help='Number of test samples (default: use Config default)'
    )
    return parser.parse_args()


def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.
    Includes safety check for dataset sizes.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)

    # Load full datasets
    harmful_train_full = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
    harmless_train_full = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)
    harmful_val_full = load_dataset_split(harmtype='harmful', split='val', instructions_only=True)
    harmless_val_full = load_dataset_split(harmtype='harmless', split='val', instructions_only=True)

    # Sample with size checks
    n_train_actual = min(cfg.n_train, len(harmful_train_full), len(harmless_train_full))
    n_val_actual = min(cfg.n_val, len(harmful_val_full), len(harmless_val_full))

    if n_train_actual < cfg.n_train:
        print(f"Warning: Requested {cfg.n_train} train samples but only {n_train_actual} available")
    if n_val_actual < cfg.n_val:
        print(f"Warning: Requested {cfg.n_val} val samples but only {n_val_actual} available")

    harmful_train = random.sample(harmful_train_full, n_train_actual)
    harmless_train = random.sample(harmless_train_full, n_train_actual)
    harmful_val = random.sample(harmful_val_full, n_val_actual)
    harmless_val = random.sample(harmless_val_full, n_val_actual)

    return harmful_train, harmless_train, harmful_val, harmless_val


def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """Filter datasets based on refusal scores."""
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(
            model_base.model, harmful_train, model_base.tokenize_instructions_fn,
            model_base.refusal_toks, tokenizer=model_base.tokenizer,
            refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
        )
        harmless_train_scores = get_refusal_scores(
            model_base.model, harmless_train, model_base.tokenize_instructions_fn,
            model_base.refusal_toks, tokenizer=model_base.tokenizer,
            refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
        )
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(
            model_base.model, harmful_val, model_base.tokenize_instructions_fn,
            model_base.refusal_toks, tokenizer=model_base.tokenizer,
            refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
        )
        harmless_val_scores = get_refusal_scores(
            model_base.model, harmless_val, model_base.tokenize_instructions_fn,
            model_base.refusal_toks, tokenizer=model_base.tokenizer,
            refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
        )
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)

    return harmful_train, harmless_train, harmful_val, harmless_val


def generate_and_save_candidate_directions(cfg, model_base, harmless_train, harmful_train):
    """Generate and save candidate directions using Arditi mean-difference method."""
    output_dir = os.path.join(cfg.artifact_path(), 'generate_directions')
    os.makedirs(output_dir, exist_ok=True)

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=output_dir
    )

    torch.save(mean_diffs, os.path.join(output_dir, 'mean_diffs.pt'))
    return mean_diffs


def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, mean_diffs):
    """Select and save the best direction using Arditi criteria."""
    output_dir = os.path.join(cfg.artifact_path(), 'select_direction')
    os.makedirs(output_dir, exist_ok=True)

    mu_b = torch.zeros(mean_diffs.size(-1))

    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        mean_diffs,
        artifact_dir=output_dir,
        coeff=1.0,
        mu_b=mu_b,
        tau=1.0
    )

    with open(os.path.join(cfg.artifact_path(), 'direction_metadata.json'), "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, os.path.join(cfg.artifact_path(), 'direction.pt'))
    torch.save(mu_b, os.path.join(cfg.artifact_path(), 'mu_b.pt'))

    return layer, pos, direction, mu_b


def generate_and_save_completions(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    output_dir = os.path.join(cfg.artifact_path(), 'completions')
    os.makedirs(output_dir, exist_ok=True)

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(
        dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=cfg.max_new_tokens
    )

    print("\n\n".join(json.dumps(c, indent=2) for c in completions[:3]))

    output_path = os.path.join(output_dir, f'{dataset_name}_{intervention_label}_completions.json')
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(completions, f, indent=4, ensure_ascii=False)

    return completions


def evaluate_and_save_results(cfg, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results."""
    completions_path = os.path.join(cfg.artifact_path(), 'completions', f'{dataset_name}_{intervention_label}_completions.json')

    with open(completions_path, 'r', encoding="utf-8") as f:
        completions = json.load(f)

    eval_path = os.path.join(cfg.artifact_path(), 'completions', f'{dataset_name}_{intervention_label}_evaluations.json')

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=eval_path,
    )

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=4, ensure_ascii=False)

    return evaluation


def save_metadata(cfg, args, layer, pos):
    """Save run metadata to JSON."""
    metadata = {
        "model_path": args.model_path,
        "intervention_type": "arditi_actadd",
        "coefficient": args.coeff,
        "system_prompt": args.system_prompt,
        "layer": layer,
        "position": pos,
        "n_train": cfg.n_train,
        "n_val": cfg.n_val,
        "n_test": cfg.n_test,
        "timestamp": datetime.now().isoformat(),
        "filter_thresholds": {
            "kl_threshold": cfg.kl_threshold,
            "steering_score_threshold": cfg.steering_score_threshold,
            "prune_layer_percentage": cfg.prune_layer_percentage
        }
    }

    with open(os.path.join(cfg.artifact_path(), 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


def run_pipeline(args):
    """Run the Arditi pipeline."""
    print("=" * 80)
    print("ARDITI REFUSAL DIRECTION PIPELINE")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Coefficient: {args.coeff}")
    print("=" * 80)

    # Setup configuration (only override if explicitly specified)
    model_alias = os.path.basename(args.model_path) + "/arditi"
    config_kwargs = {
        "model_alias": model_alias,
        "model_path": args.model_path,
        "coeff": args.coeff,
    }
    if args.system_prompt is not None:
        config_kwargs["system_prompt"] = args.system_prompt
    if args.n_train is not None:
        config_kwargs["n_train"] = args.n_train
    if args.n_val is not None:
        config_kwargs["n_val"] = args.n_val
    if args.n_test is not None:
        config_kwargs["n_test"] = args.n_test
    cfg = Config(**config_kwargs)
    print(f"System prompt: {cfg.system_prompt}")

    # Load model with system prompt from config
    print("\nLoading model...")
    model_base = construct_model_base(cfg.model_path, system_prompt=cfg.system_prompt)
    model_base.model.config.pad_token_id = model_base.tokenizer.pad_token_id
    print(f"Model loaded with system prompt: {cfg.system_prompt}")

    # Load and sample datasets
    print("\nLoading datasets...")
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)
    print(f"Raw data: {len(harmful_train)} harmful, {len(harmless_train)} harmless")

    # Filter datasets
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
        cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val
    )
    print(f"Filtered data: {len(harmful_train)} harmful, {len(harmless_train)} harmless")

    # Generate or load candidate directions
    if args.skip_generate:
        print("\nLoading cached mean_diffs...")
        mean_diffs = torch.load(os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'), map_location="cpu")
    else:
        print("\nGenerating candidate directions...")
        mean_diffs = generate_and_save_candidate_directions(cfg, model_base, harmless_train, harmful_train)

    # Select or load direction
    if args.skip_select:
        print("\nLoading cached direction...")
        meta = json.load(open(os.path.join(cfg.artifact_path(), 'direction_metadata.json'), "r"))
        layer = meta["layer"]
        pos = meta["pos"]
        mu_b = torch.load(os.path.join(cfg.artifact_path(), 'mu_b.pt'), map_location="cpu")
        direction = torch.load(os.path.join(cfg.artifact_path(), 'direction.pt'), map_location="cpu")
    else:
        print("\nSelecting best direction...")
        layer, pos, direction, mu_b = select_and_save_direction(
            cfg, model_base, harmful_val, harmless_val, mean_diffs
        )

    print(f"\nBest direction @ layer {layer}, position {pos}")

    # Prepare direction for intervention
    direction = direction.to(model_base.model.device, dtype=getattr(model_base.model, "dtype", torch.float16))
    if direction.dim() == 1:
        direction = direction.unsqueeze(-1)

    # Setup hooks
    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    actadd_fwd_pre_hooks = [(
        model_base.model_block_modules[layer],
        get_activation_addition_subspace_input_pre_hook(direction, coeff=torch.tensor(-args.coeff))
    )]
    actadd_fwd_hooks = []
    actadd_refusal_pre_hooks = [(
        model_base.model_block_modules[layer],
        get_activation_addition_subspace_input_pre_hook(direction, coeff=torch.tensor(+args.coeff))
    )]
    actadd_refusal_hooks = []

    # Load test sets
    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

    # Baseline evaluation
    if not args.skip_baseline:
        print("\n" + "-" * 80)
        print("BASELINE (No Intervention)")
        print("-" * 80)

        for dataset_name in cfg.evaluation_datasets:
            print(f"\nProcessing {dataset_name}...")
            generate_and_save_completions(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', dataset_name)
            evaluate_and_save_results(cfg, 'baseline', dataset_name, cfg.jailbreak_eval_methodologies)

        print("\nProcessing harmless test set...")
        generate_and_save_completions(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmless', dataset=harmless_test)
        evaluate_and_save_results(cfg, 'baseline', 'harmless', cfg.refusal_eval_methodologies)

    # ActAdd intervention
    print("\n" + "-" * 80)
    print(f"ACTADD INTERVENTION (coeff={-args.coeff})")
    print("-" * 80)

    for dataset_name in cfg.evaluation_datasets:
        print(f"\nProcessing {dataset_name}...")
        generate_and_save_completions(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd', dataset_name)
        evaluate_and_save_results(cfg, 'actadd', dataset_name, cfg.jailbreak_eval_methodologies)

    print("\nProcessing harmless test set (refusal induction)...")
    generate_and_save_completions(cfg, model_base, actadd_refusal_pre_hooks, actadd_refusal_hooks, 'actadd', 'harmless', dataset=harmless_test)
    evaluate_and_save_results(cfg, 'actadd', 'harmless', cfg.refusal_eval_methodologies)

    # Save metadata
    save_metadata(cfg, args, layer, pos)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {cfg.artifact_path()}")
    print(f"Selected direction: Layer {layer}, Position {pos}")


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args)
