"""
Expert-Specific Refusal Vector Pipeline

This pipeline generates and evaluates refusal vectors for individual MoE experts,
following the Arditi mean-difference approach but applied per-expert rather than per-layer.

Key differences from standard Arditi pipeline:
1. Selects candidate experts based on activation frequency differences
2. For each expert, forces that expert and extracts MLP output activations
3. Generates one candidate vector per (expert, token_position) instead of (layer, position)
4. Uses Arditi evaluation/selection to pick best expert-specific vector
5. Applies intervention weighted by expert's routing probability
"""

# Import for LoRA adapter support
try:
    from unsloth import FastLanguageModel
    from peft import PeftModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

import torch
import random
import json
import os
import argparse
from pathlib import Path

from dataset.load_dataset import load_dataset_split, load_dataset
from config import Config, SYSTEM_PROMPTS
from model_utils.model_factory_moe import construct_model_base
from submodules.arditi.select_direction import get_refusal_scores
from submodules.evaluate_jailbreak import evaluate_jailbreak

from submodules.expert_steering.expert_selection import get_candidate_experts
from submodules.expert_steering.expert_specific_activations import get_expert_mean_diff
from submodules.expert_steering.expert_intervention import get_expert_weighted_intervention_hooks
from submodules.expert_steering.select_direction_moe import select_expert_direction




def load_model_with_optional_adapter(model_path, adapter_path=None, system_prompt=None):
    """
    Load model with optional LoRA adapter.

    Args:
        model_path: Base model path
        adapter_path: Optional path to LoRA adapter
        system_prompt: System prompt option ("none", "llama_2", "lightweight") or text

    Returns:
        model_base: ModelBase object (OSSModel or wrapper)
    """
    # Resolve system prompt option to text
    system_prompt_text = SYSTEM_PROMPTS.get(system_prompt, system_prompt)

    if adapter_path is not None:
        if not UNSLOTH_AVAILABLE:
            raise ImportError("unsloth and peft are required for adapter loading. Please install them.")

        print(f"Loading base model: {model_path}")
        print(f"Loading adapter from: {adapter_path}")

        # Load with unsloth
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )

        # Load PEFT adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✓ Loaded model with LoRA adapter")

        # Wrap in OSSModel-like interface
        from model_utils.oss_model import OSSModel
        model_base = OSSModel.__new__(OSSModel)
        model_base.model = model
        model_base.tokenizer = tokenizer

        # Set up tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Set refusal tokens
        from model_utils.oss_model import OSS_REFUSAL_TOKS
        model_base.refusal_toks = OSS_REFUSAL_TOKS
        model_base.refusal_score_suffix_toks = None

        # Set end-of-instruction tokens (for plotting)
        model_base.eoi_toks = [tokenizer.encode("<|end|>", add_special_tokens=False)]

        # Set tokenization function with system prompt
        from model_utils.oss_model import tokenize_instructions_oss_chat
        model_base._system_prompt = system_prompt_text
        model_base.tokenize_instructions_fn = lambda instructions, **kwargs: tokenize_instructions_oss_chat(
            tokenizer, instructions, system=system_prompt_text, **kwargs
        )

        # Set generation function
        def generate_completions(dataset, fwd_pre_hooks=None, fwd_hooks=None, max_new_tokens=256, batch_size=8):
            """Generate completions for a dataset with optional hooks.

            Args:
                dataset: List of dicts with 'instruction' and 'category' keys
                fwd_pre_hooks: List of (module, hook_fn) tuples for forward pre-hooks
                fwd_hooks: List of (module, hook_fn) tuples for forward hooks
                max_new_tokens: Maximum tokens to generate
                batch_size: Batch size for generation

            Returns:
                List of dicts with 'prompt', 'response', and 'category' keys
            """
            completions = []

            # Extract instructions and categories from dataset
            instructions = [x['instruction'] for x in dataset]
            categories = [x['category'] for x in dataset]

            # Register hooks
            hook_handles = []
            if fwd_pre_hooks:
                for module, hook in fwd_pre_hooks:
                    handle = module.register_forward_pre_hook(hook)
                    hook_handles.append(handle)
            if fwd_hooks:
                for module, hook in fwd_hooks:
                    handle = module.register_forward_hook(hook)
                    hook_handles.append(handle)

            try:
                for i in range(0, len(instructions), batch_size):
                    batch_instructions = instructions[i:i+batch_size]
                    batch_categories = categories[i:i+batch_size]

                    tokenized = model_base.tokenize_instructions_fn(
                        batch_instructions,
                        include_trailing_whitespace=True
                    )
                    input_ids = tokenized['input_ids'].to(model.device)
                    attention_mask = tokenized['attention_mask'].to(model.device)

                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )

                    for j, output in enumerate(outputs):
                        prompt_length = input_ids[j].shape[0]
                        completion_ids = output[prompt_length:]
                        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

                        completions.append({
                            "prompt": batch_instructions[j],
                            "response": completion,
                            "category": batch_categories[j]
                        })
            finally:
                for handle in hook_handles:
                    handle.remove()

            return completions

        model_base.generate_completions = generate_completions

        return model_base
    else:
        # Standard loading with system prompt
        return construct_model_base(model_path, system_prompt=system_prompt)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Expert-specific refusal vector pipeline for MoE models"
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='unsloth/gpt-oss-20b-unsloth-bnb-4bit',
        help='Path to the base model'
    )

    parser.add_argument(
        '--adapter_path',
        type=str,
        default=None,
        help='Path to LoRA adapter checkpoint (optional, for fine-tuned models)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=15.0,
        help='Expert selection threshold (percentage points)'
    )

    parser.add_argument(
        '--expert_type',
        type=str,
        default='both',
        choices=['harmful_preferred', 'harmless_preferred', 'both'],
        help='Which type of experts to select'
    )

    parser.add_argument(
        '--skip_generate',
        action='store_true',
        help='Skip generation and load from cache'
    )

    parser.add_argument(
        '--skip_select',
        action='store_true',
        help='Skip selection and load from cache'
    )

    parser.add_argument(
        '--skip_eval',
        action='store_true',
        help='Skip evaluation (only generate and select)'
    )

    parser.add_argument(
        '--skip_baseline',
        action='store_true',
        help='Skip eval on baseline model'
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

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing'
    )

    parser.add_argument(
        '--coeff',
        type=float,
        default=1.0,
        help='Coefficient for activation addition'
    )

    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=100,
        help='Maximum new tokens for generation'
    )

    parser.add_argument(
        '--top_n',
        type=int,
        default=1,
        help='Number of top vectors to select and use for steering (default: 1)'
    )

    parser.add_argument(
        '--eval_datasets',
        type=str,
        nargs='+',
        default=None,
        help='Evaluation datasets to use (default: uses config). Options: jailbreakbench, advbench, tdc2023, maliciousinstruct, strongreject, harmbench_test'
    )

    parser.add_argument(
        '--system_prompt',
        type=str,
        default=None,
        choices=['none', 'llama_2', 'lightweight'],
        help='System prompt to use (default: use Config default)'
    )

    return parser.parse_args()


def load_and_sample_datasets(cfg):
    """Load and sample datasets with size safety checks."""
    random.seed(42)

    # Load full datasets
    harmful_train_full = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
    harmless_train_full = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)
    harmful_val_full = load_dataset_split(harmtype='harmful', split='val', instructions_only=True)
    harmless_val_full = load_dataset_split(harmtype='harmless', split='val', instructions_only=True)
    harmful_test_full = load_dataset_split(harmtype='harmful', split='test', instructions_only=True)
    harmless_test_full = load_dataset_split(harmtype='harmless', split='test', instructions_only=False)

    # Sample with size checks
    n_train_actual = min(cfg.n_train, len(harmful_train_full), len(harmless_train_full))
    n_val_actual = min(cfg.n_val, len(harmful_val_full), len(harmless_val_full))
    n_test_actual = min(cfg.n_test, len(harmful_test_full), len(harmless_test_full))

    if n_train_actual < cfg.n_train:
        print(f"Warning: Requested {cfg.n_train} train samples but only {n_train_actual} available")
    if n_val_actual < cfg.n_val:
        print(f"Warning: Requested {cfg.n_val} val samples but only {n_val_actual} available")
    if n_test_actual < cfg.n_test:
        print(f"Warning: Requested {cfg.n_test} test samples but only {n_test_actual} available")

    harmful_train = random.sample(harmful_train_full, n_train_actual)
    harmless_train = random.sample(harmless_train_full, n_train_actual)
    harmful_val = random.sample(harmful_val_full, n_val_actual)
    harmless_val = random.sample(harmless_val_full, n_val_actual)
    harmful_test = random.sample(harmful_test_full, n_test_actual)
    harmless_test = random.sample(harmless_test_full, n_test_actual)

    return (harmful_train, harmless_train, harmful_val,
            harmless_val, harmful_test, harmless_test)


def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        print("\n  Filtering training data...")
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

        harmful_train_filtered = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train_filtered = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

        print(f"    Harmful train: {len(harmful_train)} -> {len(harmful_train_filtered)}")
        print(f"    Harmless train: {len(harmless_train)} -> {len(harmless_train_filtered)}")

        harmful_train = harmful_train_filtered
        harmless_train = harmless_train_filtered

    if cfg.filter_val:
        print("\n  Filtering validation data...")
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

        harmful_val_filtered = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val_filtered = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)

        print(f"    Harmful val: {len(harmful_val)} -> {len(harmful_val_filtered)}")
        print(f"    Harmless val: {len(harmless_val)} -> {len(harmless_val_filtered)}")

        harmful_val = harmful_val_filtered
        harmless_val = harmless_val_filtered

    return harmful_train, harmless_train, harmful_val, harmless_val


def generate_expert_specific_directions(
    model_base,
    harmful_train,
    harmless_train,
    candidate_experts,
    artifact_dir,
    batch_size=32
):
    """
    Generate mean difference vectors for each candidate expert.

    Returns:
        Dictionary mapping (layer, expert_id) -> mean_diff tensor [pos, d_model]
    """
    os.makedirs(artifact_dir, exist_ok=True)

    expert_directions = {}

    print("\n" + "="*80)
    print("GENERATING EXPERT-SPECIFIC DIRECTIONS")
    print("="*80)

    for layer_idx, expert_id, diff_pct in candidate_experts:
        print(f"\n[Expert: Layer {layer_idx}, Expert {expert_id}, Diff: {diff_pct:.2f}%]")

        # Compute mean difference for this expert
        mean_diff = get_expert_mean_diff(
            model_base,
            harmful_train,
            harmless_train,
            layer_idx,
            expert_id,
            batch_size=batch_size
        )

        expert_directions[(layer_idx, expert_id)] = mean_diff

        # Save individual expert direction
        save_path = os.path.join(
            artifact_dir,
            f"expert_L{layer_idx}_E{expert_id}_mean_diff.pt"
        )
        torch.save(mean_diff, save_path)
        print(f"  Saved to: {save_path}")

    # Save combined dictionary
    combined_path = os.path.join(artifact_dir, "all_expert_directions.pt")
    torch.save(expert_directions, combined_path)
    print(f"\n✓ Saved all expert directions to: {combined_path}")

    return expert_directions


def create_candidate_tensor(expert_directions):
    """
    Convert expert directions to format compatible with Arditi's select_direction.

    The select_direction function expects: [n_positions, n_layers, d_model]
    We have: dict of {(layer, expert): [n_positions, d_model]}

    We'll create a synthetic "layer" dimension where each (layer, expert) pair
    gets its own index. Then select_direction will pick the best across all.
    """
    # Get dimensions from first entry
    first_key = list(expert_directions.keys())[0]
    first_tensor = expert_directions[first_key]
    n_positions, d_model = first_tensor.shape

    n_candidates = len(expert_directions)

    # Create tensor: [n_positions, n_candidates, d_model]
    candidates = torch.zeros(n_positions, n_candidates, d_model,
                            dtype=first_tensor.dtype, device=first_tensor.device)

    # Create mapping from candidate_idx -> (layer, expert)
    candidate_mapping = {}

    for idx, ((layer, expert), direction) in enumerate(expert_directions.items()):
        candidates[:, idx, :] = direction
        candidate_mapping[idx] = (layer, expert)

    return candidates, candidate_mapping


def select_best_expert_direction(
    model_base,
    harmful_val,
    harmless_val,
    expert_directions,
    artifact_dir,
    model_card,
    top_n=1
):
    """
    Select the best expert-specific direction(s) using Arditi's criteria.

    Returns:
        If top_n == 1: (pos, layer, expert_id, direction, mu_b)
        If top_n > 1: (selected_directions_list, mu_b) where selected_directions_list
                      contains list of (pos, layer, expert_id, direction) tuples
    """
    os.makedirs(artifact_dir, exist_ok=True)

    print("\n" + "="*80)
    if top_n == 1:
        print("SELECTING BEST EXPERT DIRECTION")
    else:
        print(f"SELECTING TOP {top_n} EXPERT DIRECTIONS")
    print("="*80)

    # Convert to format compatible with select_direction
    candidates, candidate_mapping = create_candidate_tensor(expert_directions)

    print(f"Candidates tensor shape: {candidates.shape}")
    print(f"Number of expert candidates: {len(candidate_mapping)}")

    # Use expert-specific MLP-level selection
    mu_b = torch.zeros(candidates.size(-1), device=candidates.device)

    result = select_expert_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidates,
        candidate_mapping,  # Pass the mapping so we know which layer each expert is in
        artifact_dir=artifact_dir,
        coeff=args.coeff,  # Use command-line argument
        mu_b=mu_b,
        tau=1.0,
        top_n=top_n,
        model_card=model_card
    )

    # Handle single vs multiple directions
    if top_n == 1:
        pos, candidate_idx, direction = result
        # Map candidate_idx back to (layer, expert)
        layer, expert_id = candidate_mapping[candidate_idx]

        print(f"\n✓ Best direction:")
        print(f"  Position: {pos}")
        print(f"  Layer: {layer}")
        print(f"  Expert: {expert_id}")
        print(f"  Direction norm: {direction.norm().item():.4f}")

        return pos, layer, expert_id, direction, mu_b
    else:
        # result is a list of (pos, candidate_idx, direction) tuples
        selected_directions = []
        for pos, candidate_idx, direction in result:
            layer, expert_id = candidate_mapping[candidate_idx]
            selected_directions.append((pos, layer, expert_id, direction))

        return selected_directions, mu_b


def generate_and_evaluate_completions(
    model_base,
    dataset_name,
    expert_info,  # Can be (layer, expert_id, direction) or list of (layer, expert_id, direction)
    coeff,
    output_dir,
    intervention_label,
    eval_methodologies,
    max_new_tokens=256,
    dataset=None
):
    """Generate completions with expert-specific intervention and evaluate.

    Args:
        expert_info: Either a single (layer, expert_id, direction) tuple for single-vector steering,
                     or a list of such tuples for multi-vector steering
    """

    # Create output directory
    completions_dir = os.path.join(output_dir, 'completions', intervention_label)
    os.makedirs(completions_dir, exist_ok=True)

    # Load dataset if not provided
    if dataset is None:
        dataset = load_dataset(dataset_name)

    # Get intervention hooks - handle both single and multiple directions
    if isinstance(expert_info, list):
        # Multiple directions
        from submodules.expert_steering.expert_intervention import get_multi_expert_weighted_intervention_hooks
        fwd_pre_hooks, fwd_hooks = get_multi_expert_weighted_intervention_hooks(
            model_base,
            expert_directions=expert_info,  # List of (layer, expert_id, direction)
            coeff=coeff
        )
    else:
        # Single direction (backward compatibility)
        layer, expert_id, direction = expert_info
        fwd_pre_hooks, fwd_hooks = get_expert_weighted_intervention_hooks(
            model_base,
            layer_idx=layer,
            expert_id=expert_id,
            direction=direction,
            coeff=coeff
        )

    # Generate completions
    print(f"\nGenerating completions for {dataset_name} with {intervention_label}...")
    completions = model_base.generate_completions(
        dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=max_new_tokens
    )

    # Save completions
    completions_path = os.path.join(
        completions_dir,
        f"{dataset_name}_completions.json"
    )
    with open(completions_path, "w", encoding="utf-8") as f:
        json.dump(completions, f, indent=4, ensure_ascii=False)

    print(f"Saved completions to: {completions_path}")

    # Evaluate
    print(f"Evaluating completions...")
    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(completions_dir, f"{dataset_name}_evaluations.json"),
    )

    # Save evaluations
    eval_path = os.path.join(completions_dir, f"{dataset_name}_evaluations.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=4, ensure_ascii=False)

    print(f"Saved evaluations to: {eval_path}")

    return completions, evaluation


def run_expert_specific_pipeline(args):
    """Run the full expert-specific pipeline."""

    print("="*80)
    print("EXPERT-SPECIFIC REFUSAL VECTOR PIPELINE")
    print("="*80)
    print(f"Model: {args.model_path}")
    if args.adapter_path:
        print(f"Adapter: {args.adapter_path}")
    print(f"Expert threshold: {args.threshold}%")
    print(f"Coefficient: {args.coeff}")
    print(f"Expert type: {args.expert_type}")
    print("="*80)

    # Setup paths
    if args.adapter_path:
        # Use adapter name in alias
        adapter_name = Path(args.adapter_path).parent.name + "_" + Path(args.adapter_path).name
        model_alias = f"{os.path.basename(args.model_path)}/expert_steering_t{args.threshold}_{adapter_name}"
    else:
        model_alias = f"{os.path.basename(args.model_path)}/expert_steering_t{args.threshold}"

    # Add top_n to alias if not default
    if args.top_n > 1:
        model_alias += f"_top{args.top_n}"

    # Create config with CLI args (only override if explicitly specified)
    config_kwargs = {
        "model_alias": model_alias,
        "model_path": args.model_path,
        "coeff": args.coeff,
        "threshold": args.threshold,
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

    # Override evaluation datasets if specified via CLI
    if args.eval_datasets is not None:
        cfg.evaluation_datasets = tuple(args.eval_datasets)
        print(f"Using custom evaluation datasets: {cfg.evaluation_datasets}")

    # Create output directory
    base_output_dir = os.path.join(cfg.artifact_path(), f"coeff_{args.coeff}")

    # If skip_select and top_n > 1, use subdirectory for this specific top_n evaluation
    if args.skip_select and args.top_n > 1:
        output_dir = os.path.join(base_output_dir, f"top_{args.top_n}_eval")
    else:
        output_dir = base_output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Load model (with optional adapter)
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model_base = load_model_with_optional_adapter(args.model_path, args.adapter_path, system_prompt=cfg.system_prompt)

    # Create model card for MoE-specific operations
    from model_utils.model_card_factory import create_model_card
    model_card = create_model_card(model_base)
    print(f"Model card: {type(model_card).__name__}")

    # Load datasets
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    (harmful_train, harmless_train, harmful_val,
     harmless_val, harmful_test, harmless_test) = load_and_sample_datasets(cfg)

    print(f"Raw data:")
    print(f"  Training: {len(harmful_train)} harmful, {len(harmless_train)} harmless")
    print(f"  Validation: {len(harmful_val)} harmful, {len(harmless_val)} harmless")
    print(f"  Test: {len(harmful_test)} harmful, {len(harmless_test)} harmless")

    # Filter datasets based on refusal scores
    print("\nFiltering datasets based on refusal scores...")
    (harmful_train, harmless_train,
     harmful_val, harmless_val) = filter_data(
        cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val
    )

    print(f"\nFiltered data:")
    print(f"  Training: {len(harmful_train)} harmful, {len(harmless_train)} harmless")
    print(f"  Validation: {len(harmful_val)} harmful, {len(harmless_val)} harmless")
    print(f"  Test: {len(harmful_test)} harmful, {len(harmless_test)} harmless")

    # Select candidate experts
    print("\n" + "="*80)
    print("SELECTING CANDIDATE EXPERTS")
    print("="*80)

    # Get model-specific expert diffs path
    expert_diffs_filename = model_card.get_expert_diffs_filename()
    expert_diffs_path = f"expert_diffs/{expert_diffs_filename}"

    # Generate expert diffs if they don't exist
    if not os.path.exists(expert_diffs_path):
        print(f"Expert diffs not found at {expert_diffs_path}, generating...")
        os.makedirs("expert_diffs", exist_ok=True)
        model_card.generate_expert_diffs(
            harmful_dataset_path="dataset/splits/harmful_train.json",
            harmless_dataset_path="dataset/splits/harmless_train.json",
            output_path=expert_diffs_path,
            batch_size=args.batch_size
        )
        print(f"Expert diffs saved to {expert_diffs_path}")

    candidate_experts = get_candidate_experts(
        threshold=args.threshold,
        expert_type=args.expert_type,
        expert_diffs_path=expert_diffs_path
    )

    # Save candidate expert info
    expert_info_path = os.path.join(output_dir, "candidate_experts.json")
    with open(expert_info_path, 'w') as f:
        expert_info = [
            {"layer": int(layer), "expert": int(expert), "diff_pct": float(diff)}
            for layer, expert, diff in candidate_experts
        ]
        json.dump(expert_info, f, indent=2)

    # Generate expert-specific directions
    if not args.skip_generate:
        expert_directions = generate_expert_specific_directions(
            model_base,
            harmful_train,
            harmless_train,
            candidate_experts,
            artifact_dir=os.path.join(output_dir, "expert_directions"),
            batch_size=args.batch_size
        )
    else:
        print("\nSkipping generation, loading from cache...")
        cache_dir = base_output_dir if args.skip_select else output_dir
        directions_path = os.path.join(cache_dir, "expert_directions", "all_expert_directions.pt")
        expert_directions = torch.load(directions_path)


    # Select best direction(s)
    if not args.skip_select:
        selection_result = select_best_expert_direction(
            model_base,
            harmful_val,
            harmless_val,
            expert_directions,
            artifact_dir=os.path.join(output_dir, "selection"),
            top_n=args.top_n,
            model_card=model_card
        )

        # Handle single vs multiple directions
        if args.top_n == 1:
            pos, layer, expert_id, direction, mu_b = selection_result

            # Save selected direction (single)
            metadata = {
                "top_n": 1,
                "pos": int(pos),
                "layer": int(layer),
                "expert_id": int(expert_id),
                "threshold": args.threshold,
                "expert_type": args.expert_type
            }

            with open(os.path.join(output_dir, "direction_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)

            torch.save(direction, os.path.join(output_dir, "direction.pt"))
            torch.save(mu_b, os.path.join(output_dir, "mu_b.pt"))

        else:
            selected_directions, mu_b = selection_result

            # Save selected directions (multiple)
            metadata = {
                "top_n": args.top_n,
                "threshold": args.threshold,
                "expert_type": args.expert_type,
                "directions": [
                    {
                        "pos": int(pos),
                        "layer": int(layer),
                        "expert_id": int(expert_id)
                    }
                    for pos, layer, expert_id, _ in selected_directions
                ]
            }

            with open(os.path.join(output_dir, "direction_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save all directions
            directions_dict = {
                f"direction_{i}": direction
                for i, (_, _, _, direction) in enumerate(selected_directions)
            }
            torch.save(directions_dict, os.path.join(output_dir, "directions.pt"))
            torch.save(mu_b, os.path.join(output_dir, "mu_b.pt"))

    else:
        print("\nSkipping selection, loading from cache...")

        # Load from base_output_dir (where the cached data is)
        cache_dir = base_output_dir if args.skip_select else output_dir

        # Load filtered evaluations JSON
        evaluations_path = os.path.join(cache_dir, "selection", "expert_direction_evaluations_filtered.json")
        with open(evaluations_path, 'r') as f:
            filtered_evals = json.load(f)

        # Load all expert directions
        all_directions_path = os.path.join(cache_dir, "expert_directions", "all_expert_directions.pt")
        all_expert_directions = torch.load(all_directions_path)

        # Load mu_b
        mu_b = torch.zeros(list(all_expert_directions.values())[0].size(-1))

        # Take top n from filtered evaluations
        n_to_load = min(args.top_n, len(filtered_evals))

        if n_to_load < args.top_n:
            print(f"Warning: Requested top_n={args.top_n} but only {len(filtered_evals)} directions available")

        if args.top_n == 1:
            # Single direction
            entry = filtered_evals[0]
            pos = entry["position"]
            layer = entry["layer"]
            expert_id = entry["expert"]

            # Load full direction tensor and extract the specific position
            full_direction = all_expert_directions[(layer, expert_id)]
            direction = full_direction[pos]

            print(f"\nLoaded direction from JSON:")
            print(f"  Position: {pos}")
            print(f"  Layer: {layer}")
            print(f"  Expert: {expert_id}")
            print(f"  Refusal score: {entry['refusal_score']:.4f}")
            print(f"  Steering score: {entry['steering_score']:.4f}")
            print(f"  KL Divergence: {entry['kl_div_score']:.4f}")

        else:
            # Multiple directions
            selected_directions = []

            print(f"\nLoading top {n_to_load} directions from JSON:")
            for i in range(n_to_load):
                entry = filtered_evals[i]
                pos = entry["position"]
                layer = entry["layer"]
                expert_id = entry["expert"]

                # Load full direction tensor and extract the specific position
                full_direction = all_expert_directions[(layer, expert_id)]
                direction = full_direction[pos]

                selected_directions.append((pos, layer, expert_id, direction))

                print(f"\n  [{i+1}/{n_to_load}]")
                print(f"    Position: {pos}")
                print(f"    Layer: {layer}")
                print(f"    Expert: {expert_id}")
                print(f"    Refusal score: {entry['refusal_score']:.4f}")
                print(f"    Steering score: {entry['steering_score']:.4f}")
                print(f"    KL Divergence: {entry['kl_div_score']:.4f}")

    # Evaluation
    if not args.skip_eval:
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80)

        # Prepare expert_info for evaluation - handle single vs multiple
        if args.top_n == 1:
            # Move direction to model device and dtype
            direction = direction.to(model_base.model.device, dtype=model_base.model.dtype)
            expert_info = (layer, expert_id, direction)
        else:
            # Move all directions to model device and dtype
            expert_info = [
                (layer, expert_id, direction.to(model_base.model.device, dtype=model_base.model.dtype))
                for _, layer, expert_id, direction in selected_directions
            ]
        if not args.skip_baseline:
            # Baseline (no intervention)
            print("\n" + "-"*80)
            print("BASELINE (No Intervention)")
            print("-"*80)
    
            # Harmful test set
            for dataset_name in cfg.evaluation_datasets:
                generate_and_evaluate_completions(
                    model_base,
                    dataset_name,
                    expert_info,
                    coeff=0.0,  # No intervention
                    output_dir=output_dir,
                    intervention_label='baseline',
                    eval_methodologies=cfg.jailbreak_eval_methodologies,
                    max_new_tokens=args.max_new_tokens
                )
    
            # Harmless test set
            generate_and_evaluate_completions(
                model_base,
                'harmless',
                expert_info,
                coeff=0.0,
                output_dir=output_dir,
                intervention_label='baseline',
                eval_methodologies=cfg.refusal_eval_methodologies,
                max_new_tokens=args.max_new_tokens,
                dataset=harmless_test
            )

        # ActAdd intervention (suppress refusal)
        print("\n" + "-"*80)
        print(f"ACTADD INTERVENTION (coeff={-args.coeff})")
        print("-"*80)

        # Harmful test set
        for dataset_name in cfg.evaluation_datasets:
            generate_and_evaluate_completions(
                model_base,
                dataset_name,
                expert_info,
                coeff=-args.coeff,  # Negative to suppress refusal
                output_dir=output_dir,
                intervention_label='actadd',
                eval_methodologies=cfg.jailbreak_eval_methodologies,
                max_new_tokens=args.max_new_tokens
            )

        # Harmless test set (with positive coeff to induce refusal)
        generate_and_evaluate_completions(
            model_base,
            'harmless',
            expert_info,
            coeff=args.coeff,  # Positive to induce refusal
            output_dir=output_dir,
            intervention_label='actadd',
            eval_methodologies=cfg.refusal_eval_methodologies,
            max_new_tokens=args.max_new_tokens,
            dataset=harmless_test
        )

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")

    if args.top_n == 1:
        print(f"\nSelected expert-specific direction:")
        print(f"  Layer: {layer}")
        print(f"  Expert: {expert_id}")
        print(f"  Position: {pos}")
    else:
        print(f"\nSelected top {args.top_n} expert-specific directions:")
        for i, (pos, layer, expert_id, _) in enumerate(selected_directions):
            print(f"  [{i+1}] Layer: {layer}, Expert: {expert_id}, Position: {pos}")


if __name__ == "__main__":
    args = parse_arguments()
    run_expert_specific_pipeline(args)
