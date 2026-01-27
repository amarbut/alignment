"""
Expert Diff Generator Utility.

Generates expert activation frequency differences between harmful and harmless prompts.
Works with any MoE model via the ModelCard abstraction.

Usage:
    # From command line
    python expert_diff_generator.py --model_path unsloth/gpt-oss-20b-unsloth-bnb-4bit

    # From code (via model card)
    model_card.generate_expert_diffs()
"""

import os
import sys
import json
import random
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
from collections import defaultdict

if TYPE_CHECKING:
    from pipeline.model_utils.model_base import ModelBase
    from pipeline.model_utils.model_card import ModelCard

# Global storage for router outputs (used by hooks)
router_outputs = {}


def create_mlp_hook(layer_idx: int, model_card: "ModelCard"):
    """
    Create a hook to capture router logits from MLP output.

    Uses model_card.parse_mlp_output() to handle different output formats.
    """
    def hook(module, input, output):
        global router_outputs
        # Use model card to parse output (handles tuple vs tensor)
        hidden_states, router_logits = model_card.parse_mlp_output(output)

        if router_logits is not None:
            router_outputs[layer_idx] = router_logits.detach().cpu()

    return hook


def load_dataset(path: str) -> List[Dict]:
    """Load a dataset from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def extract_expert_routing(
    model_base: "ModelBase",
    model_card: "ModelCard",
    prompts: List[str],
    label: str,
    batch_size: int = 4,
    last_n_tokens: int = 5
) -> Dict[str, Dict[int, List[int]]]:
    """
    Extract expert routing information for a set of prompts.

    Args:
        model_base: ModelBase instance (for tokenization)
        model_card: ModelCard instance (for MLP access)
        prompts: List of prompt strings
        label: Label for this dataset ('harmful' or 'harmless')
        batch_size: Batch size for processing
        last_n_tokens: Number of tokens from the end to analyze

    Returns:
        Dictionary mapping layer names to lists of expert activations
        Format: {f"layer_{i}": {expert_id: [activation_count_per_prompt]}}
    """
    global router_outputs

    model = model_base.model
    tokenizer = model_base.tokenizer
    tokenize_fn = model_base.tokenize_instructions_fn

    # Track expert activations per layer
    # Structure: {layer_idx: {expert_id: count}}
    layer_expert_counts = defaultdict(lambda: defaultdict(int))
    total_tokens_analyzed = 0

    # Determine hook strategy based on model type
    use_router_hooks = model_card.uses_router_hook_for_routing()
    router_output_format = model_card.get_router_output_format() if use_router_hooks else "logits"

    # Register hooks on all MoE layers
    hooks = []
    for layer_idx in range(model_card.get_num_layers()):
        if model_card.is_moe_layer(layer_idx):
            if use_router_hooks:
                # Hook router directly (for DeepSeek-style models)
                router = model_card.get_router(layer_idx)
                hook = router.register_forward_hook(
                    model_card.create_router_hook(layer_idx, router_outputs)
                )
            else:
                # Hook MLP (for OSS/Mixtral-style models)
                mlp = model_card.get_mlp_module(layer_idx)
                hook = mlp.register_forward_hook(create_mlp_hook(layer_idx, model_card))
            hooks.append(hook)

    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Processing {label} prompts"):
            batch_prompts = prompts[i:i + batch_size]

            # Tokenize using model_base's tokenize function
            # Use keyword argument to avoid conflict with partial's bound tokenizer
            tokenized = tokenize_fn(instructions=batch_prompts)

            input_ids = tokenized.input_ids.to(model.device)
            attention_mask = tokenized.attention_mask.to(model.device)

            # Clear previous router outputs
            router_outputs.clear()

            # Forward pass (hooks capture router outputs)
            try:
                with torch.no_grad():
                    _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
            except (RuntimeError, TypeError) as e:
                # Model may throw errors but hooks still capture data
                if len(router_outputs) == 0:
                    raise
                # Continue if we got data

            # Process each sample in the batch
            for batch_idx in range(len(batch_prompts)):
                # Get sequence length for this sample
                seq_len = attention_mask[batch_idx].sum().item()

                # Focus on the last N tokens
                start_pos = max(0, int(seq_len) - last_n_tokens)
                end_pos = int(seq_len)
                num_positions = end_pos - start_pos

                # Extract routing for each MoE layer
                for layer_idx in sorted(router_outputs.keys()):
                    if not model_card.is_moe_layer(layer_idx):
                        continue

                    if router_output_format == "top_k_indices":
                        # DeepSeek-style: router_outputs[layer_idx] = {'indices': ..., 'weights': ...}
                        routing_data = router_outputs[layer_idx]
                        top_k_indices = routing_data['indices']  # [batch*seq, k]

                        batch_size_cur = len(batch_prompts)
                        seq_len_padded = input_ids.shape[1]
                        k = top_k_indices.shape[-1]

                        # Reshape to (batch_size, seq_len, k)
                        indices_reshaped = top_k_indices.view(batch_size_cur, seq_len_padded, k)

                        # Get indices for this sample's analyzed positions
                        sample_indices = indices_reshaped[batch_idx, start_pos:end_pos, :]  # [num_pos, k]

                        # Count each expert that appears in top-k (count top-1 only for consistency)
                        top_experts = sample_indices[:, 0].cpu().numpy()  # Just top-1
                        for expert_id in top_experts:
                            layer_expert_counts[layer_idx][int(expert_id)] += 1

                    else:
                        # OSS/Mixtral-style: router_outputs[layer_idx] = logits tensor
                        router_logits_flat = router_outputs[layer_idx]
                        batch_size_cur = len(batch_prompts)
                        seq_len_padded = input_ids.shape[1]
                        num_experts = router_logits_flat.shape[-1]

                        # Reshape to (batch_size, seq_len, num_experts)
                        router_logits = router_logits_flat.view(batch_size_cur, seq_len_padded, num_experts)

                        # Get logits for this sample's analyzed positions
                        layer_router = router_logits[batch_idx, start_pos:end_pos, :]

                        # Get top expert for each token
                        top_experts = torch.argmax(layer_router, dim=-1).cpu().numpy()

                        # Count activations
                        for expert_id in top_experts:
                            layer_expert_counts[layer_idx][int(expert_id)] += 1

                total_tokens_analyzed += num_positions

    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    # Convert to percentages
    layer_expert_percentages = {}
    for layer_idx in sorted(layer_expert_counts.keys()):
        expert_counts = layer_expert_counts[layer_idx]
        num_experts = model_card.get_num_experts(layer_idx)

        percentages = {}
        for expert_id in range(num_experts):
            count = expert_counts.get(expert_id, 0)
            # Percentage of total analyzed tokens where this expert was top-1
            pct = (count / total_tokens_analyzed * 100) if total_tokens_analyzed > 0 else 0
            percentages[expert_id] = pct

        layer_expert_percentages[f"layer_{layer_idx}"] = percentages

    return layer_expert_percentages


def compute_expert_diffs(
    harmful_routing: Dict[str, Dict[int, float]],
    harmless_routing: Dict[str, Dict[int, float]]
) -> Dict[str, List[List]]:
    """
    Compute expert activation frequency differences (harmful - harmless).

    Args:
        harmful_routing: Expert percentages for harmful prompts
        harmless_routing: Expert percentages for harmless prompts

    Returns:
        Dictionary mapping layer names to [[expert_id, diff_percentage], ...]
    """
    expert_diffs = {}

    for layer_name in harmful_routing:
        if layer_name not in harmless_routing:
            continue

        harmful_pcts = harmful_routing[layer_name]
        harmless_pcts = harmless_routing[layer_name]

        # Get all expert IDs
        all_experts = set(harmful_pcts.keys()) | set(harmless_pcts.keys())

        diffs = []
        for expert_id in sorted(all_experts):
            harmful_pct = harmful_pcts.get(expert_id, 0)
            harmless_pct = harmless_pcts.get(expert_id, 0)
            diff = harmful_pct - harmless_pct
            diffs.append([expert_id, diff])

        expert_diffs[layer_name] = diffs

    return expert_diffs


def generate_expert_diffs_for_model(
    model_base: "ModelBase",
    model_card: "ModelCard",
    harmful_dataset_path: str = "dataset/splits/harmful_train.json",
    harmless_dataset_path: str = "dataset/splits/harmless_train.json",
    output_path: Optional[str] = None,
    batch_size: int = 4,
    last_n_tokens: int = 5,
    num_harmful: Optional[int] = None,
    num_harmless: int = 200
) -> Dict:
    """
    Generate expert activation frequency differences for any MoE model.

    This is the main entry point called by ModelCard.generate_expert_diffs().

    Args:
        model_base: ModelBase instance
        model_card: ModelCard instance
        harmful_dataset_path: Path to harmful prompts JSON
        harmless_dataset_path: Path to harmless prompts JSON
        output_path: Where to save results (default: expert_explore/{filename})
        batch_size: Batch size for processing
        last_n_tokens: Number of tokens from end to analyze
        num_harmful: Number of harmful samples (None = all)
        num_harmless: Number of harmless samples

    Returns:
        Dictionary with expert activation frequency differences
    """
    print("="*80)
    print("EXPERT DIFF GENERATOR")
    print(f"Model: {model_base.model_name_or_path}")
    print(f"MLP output format: {model_card._mlp_output_format}")
    print("="*80)

    # Determine output path
    if output_path is None:
        output_path = f"expert_explore/{model_card.get_expert_diffs_filename()}"

    # Load datasets
    print("\nLoading datasets...")
    harmful_data = load_dataset(harmful_dataset_path)
    harmless_data = load_dataset(harmless_dataset_path)

    # Extract prompts
    harmful_prompts = [item["instruction"] for item in harmful_data]
    if num_harmful is not None:
        harmful_prompts = harmful_prompts[:num_harmful]

    harmless_prompts = [item["instruction"] for item in random.sample(harmless_data, min(num_harmless, len(harmless_data)))]

    print(f"Harmful prompts: {len(harmful_prompts)}")
    print(f"Harmless prompts: {len(harmless_prompts)}")
    print(f"Last N tokens: {last_n_tokens}")

    # Model info
    num_layers = model_card.get_num_layers()
    moe_layers = sum(1 for i in range(num_layers) if model_card.is_moe_layer(i))
    num_experts = model_card.get_num_experts(0 if model_card.is_moe_layer(0) else 1)

    print(f"\nModel info:")
    print(f"  Total layers: {num_layers}")
    print(f"  MoE layers: {moe_layers}")
    print(f"  Experts per MoE layer: {num_experts}")
    print(f"  Routing mode: {model_card.get_expert_routing_mode()}")

    # Extract routing for harmful prompts
    print("\n" + "="*80)
    print("EXTRACTING ROUTING FOR HARMFUL PROMPTS")
    print("="*80)

    harmful_routing = extract_expert_routing(
        model_base=model_base,
        model_card=model_card,
        prompts=harmful_prompts,
        label="harmful",
        batch_size=batch_size,
        last_n_tokens=last_n_tokens
    )

    # Extract routing for harmless prompts
    print("\n" + "="*80)
    print("EXTRACTING ROUTING FOR HARMLESS PROMPTS")
    print("="*80)

    harmless_routing = extract_expert_routing(
        model_base=model_base,
        model_card=model_card,
        prompts=harmless_prompts,
        label="harmless",
        batch_size=batch_size,
        last_n_tokens=last_n_tokens
    )

    # Compute differences
    print("\n" + "="*80)
    print("COMPUTING EXPERT DIFFS")
    print("="*80)

    expert_diffs = compute_expert_diffs(harmful_routing, harmless_routing)

    # Prepare results
    results = {
        "model_path": model_base.model_name_or_path,
        "harmful_dataset": harmful_dataset_path,
        "harmless_dataset": harmless_dataset_path,
        "num_harmful": len(harmful_prompts),
        "num_harmless": len(harmless_prompts),
        "last_n_tokens": last_n_tokens,
        "num_layers": num_layers,
        "num_moe_layers": moe_layers,
        "num_experts": num_experts,
        "routing_mode": model_card.get_expert_routing_mode(),
        "expert_diffs": expert_diffs
    }

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print summary of most different experts
    print("\n" + "="*80)
    print("TOP EXPERTS BY HARMFUL/HARMLESS DIFFERENCE")
    print("="*80)

    all_diffs = []
    for layer_name, diffs in expert_diffs.items():
        for expert_id, diff_pct in diffs:
            all_diffs.append((layer_name, expert_id, diff_pct))

    # Sort by absolute difference
    all_diffs.sort(key=lambda x: abs(x[2]), reverse=True)

    print("\nTop 10 experts with highest difference:")
    for layer_name, expert_id, diff_pct in all_diffs[:10]:
        direction = "more harmful" if diff_pct > 0 else "more harmless"
        print(f"  {layer_name}, expert {expert_id}: {diff_pct:+.2f}% ({direction})")

    return results


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate expert activation frequency differences for MoE models"
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='unsloth/gpt-oss-20b-unsloth-bnb-4bit',
        help='Model path to load'
    )
    parser.add_argument(
        '--harmful_dataset',
        type=str,
        default='dataset/splits/harmful_train.json',
        help='Path to harmful dataset'
    )
    parser.add_argument(
        '--harmless_dataset',
        type=str,
        default='dataset/splits/harmless_train.json',
        help='Path to harmless dataset'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Output path for results (default: expert_explore/{model}_expert_diffs.json)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--last_n_tokens',
        type=int,
        default=5,
        help='Number of tokens from end to analyze'
    )
    parser.add_argument(
        '--num_harmful',
        type=int,
        default=None,
        help='Number of harmful samples (None = all)'
    )
    parser.add_argument(
        '--num_harmless',
        type=int,
        default=200,
        help='Number of harmless samples'
    )

    args = parser.parse_args()

    # Import here to avoid circular imports
    from pipeline.model_utils.model_factory import construct_model_base
    from pipeline.model_utils.model_card_factory import create_model_card

    # Load model and create model card
    print(f"Loading model: {args.model_path}")
    model_base = construct_model_base(args.model_path)
    model_card = create_model_card(model_base)

    # Generate expert diffs
    generate_expert_diffs_for_model(
        model_base=model_base,
        model_card=model_card,
        harmful_dataset_path=args.harmful_dataset,
        harmless_dataset_path=args.harmless_dataset,
        output_path=args.output_path,
        batch_size=args.batch_size,
        last_n_tokens=args.last_n_tokens,
        num_harmful=args.num_harmful,
        num_harmless=args.num_harmless
    )


if __name__ == "__main__":
    main()
