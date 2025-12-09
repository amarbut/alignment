"""
Extract expert routing patterns during response generation using BeaverTails dataset.

Based on Fayyaz et al. (2025), this script analyzes expert selection patterns during
actual refusal vs. compliance responses using matched pairs from BeaverTails.

Key differences from previous extraction:
1. Uses BeaverTails dataset with matched pairs (same prompt, different responses)
2. Extracts routing during RESPONSE tokens, not just prompt
3. Compares refusal tokens vs. harmful response tokens

Uses the MLP output workaround from v2 to capture router logits.
"""

import os
import sys
import torch
import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from typing import List, Dict, Tuple
import numpy as np

# Add alignment directory to path
alignment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if alignment_dir not in sys.path:
    sys.path.insert(0, alignment_dir)

from pipeline.model_utils.model_factory import construct_model_base

# Global storage for router outputs
router_outputs = {}


def create_mlp_hook(layer_idx):
    """
    Create a hook to capture router logits from MLP output.

    This uses the same workaround as extract_expert_routing_v2.py.
    The MLP output is a tuple: (hidden_states, router_logits).
    """
    def hook(module, input, output):
        # MLP output is a tuple: (hidden_states, router_logits)
        # router_logits shape: (batch_size * seq_len, num_experts)
        if isinstance(output, tuple) and len(output) >= 2:
            router_logits = output[1]  # Second element is router logits
            router_outputs[layer_idx] = router_logits.detach().cpu()
    return hook


def load_beavertails_matched_pairs(n_pairs: int = 100, split: str = '30k_test'):
    """
    Load BeaverTails dataset and create matched pairs.

    BeaverTails has:
    - 'prompt': The user request
    - 'response': The model response
    - 'is_safe': Whether the response is safe (True) or harmful (False)

    We want matched pairs where the same prompt has both a safe (refusing) and
    unsafe (complying) response.

    Args:
        n_pairs: Number of matched pairs to extract
        split: Which BeaverTails split to use

    Returns:
        List of matched pairs, each containing:
        {
            'prompt': str,
            'refusal_response': str,
            'harmful_response': str
        }
    """
    print(f"Loading BeaverTails dataset (split: {split})...")

    # Load BeaverTails from HuggingFace
    dataset = load_dataset("PKU-Alignment/BeaverTails", split=split)

    print(f"Loaded {len(dataset)} examples")

    # Group by prompt
    prompt_groups = {}
    for item in tqdm(dataset, desc="Grouping by prompt"):
        prompt = item['prompt']
        is_safe = item['is_safe']
        response = item['response']

        if prompt not in prompt_groups:
            prompt_groups[prompt] = {'safe': [], 'unsafe': []}

        if is_safe:
            prompt_groups[prompt]['safe'].append(response)
        else:
            prompt_groups[prompt]['unsafe'].append(response)

    # Find matched pairs (prompts with both safe and unsafe responses)
    matched_pairs = []
    for prompt, responses in prompt_groups.items():
        if responses['safe'] and responses['unsafe']:
            # Take first response of each type
            matched_pairs.append({
                'prompt': prompt,
                'refusal_response': responses['safe'][0],
                'harmful_response': responses['unsafe'][0]
            })

        if len(matched_pairs) >= n_pairs:
            break

    print(f"\nFound {len(matched_pairs)} matched pairs (prompts with both safe and unsafe responses)")

    if len(matched_pairs) == 0:
        print("\nWARNING: No matched pairs found!")
        print("This might mean the dataset doesn't have prompts with multiple responses.")
        print("Falling back to separate safe/unsafe examples...")

        # Fallback: Just take separate examples
        safe_examples = [item for item in dataset if item['is_safe']][:n_pairs]
        unsafe_examples = [item for item in dataset if not item['is_safe']][:n_pairs]

        matched_pairs = [
            {
                'prompt': safe['prompt'],
                'refusal_response': safe['response'],
                'harmful_response': unsafe['response']
            }
            for safe, unsafe in zip(safe_examples, unsafe_examples)
        ]

    return matched_pairs[:n_pairs]


def extract_routing_for_response(
    model_base,
    prompt: str,
    response: str,
    label: str,
    max_response_tokens: int = 50
) -> Dict:
    """
    Extract expert routing patterns for response tokens.

    Uses teacher forcing: provides the full sequence (prompt + response) and
    extracts routing for the response portion only.

    Args:
        model_base: The model wrapper
        prompt: The input prompt
        response: The response to analyze
        label: 'refusal' or 'harmful'
        max_response_tokens: Maximum response tokens to analyze

    Returns:
        Dictionary with routing information for response tokens
    """
    global router_outputs

    # Tokenize prompt and full sequence separately
    prompt_tokenized = model_base.tokenize_instructions_fn(instructions=[prompt])
    prompt_ids = prompt_tokenized.input_ids.to(model_base.model.device)
    prompt_len = prompt_ids.shape[1]

    # Tokenize full sequence (prompt + response)
    # We need to manually construct this to know where response starts
    full_text_formatted = model_base.tokenize_instructions_fn(
        instructions=[prompt]
    )

    # Get the formatted prompt
    formatted_prompt = model_base.tokenizer.decode(
        full_text_formatted.input_ids[0],
        skip_special_tokens=False
    )

    # Add response to formatted prompt
    full_text = formatted_prompt + response

    # Tokenize the full sequence
    full_tokenized = model_base.tokenizer(
        full_text,
        return_tensors="pt",
        padding=False,
        truncation=False
    )

    full_ids = full_tokenized.input_ids.to(model_base.model.device)
    attention_mask = torch.ones_like(full_ids)

    # Determine response token positions
    # Response starts after the prompt
    response_start = prompt_len
    response_end = min(full_ids.shape[1], prompt_len + max_response_tokens)
    response_len = response_end - response_start

    if response_len <= 0:
        return None

    # Register hooks on all MLPs
    hooks = []
    for layer_idx, layer in enumerate(model_base.model.model.layers):
        hook = layer.mlp.register_forward_hook(create_mlp_hook(layer_idx))
        hooks.append(hook)

    router_outputs = {}

    try:
        # Forward pass through full sequence
        with torch.no_grad():
            _ = model_base.model(
                input_ids=full_ids,
                attention_mask=attention_mask
            )
    except (RuntimeError, TypeError) as e:
        if len(router_outputs) == 0:
            print(f"Error during forward pass: {e}")
            return None
        # Continue if we got router outputs despite error
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    # Process router outputs - extract response tokens only
    layer_routing = {}

    for layer_idx in sorted(router_outputs.keys()):
        # Router logits shape: (batch_size * seq_len, num_experts)
        # Need to reshape to (batch_size, seq_len, num_experts)
        router_logits_flat = router_outputs[layer_idx]

        batch_size = 1  # We process one at a time
        seq_len = full_ids.shape[1]
        num_experts = router_logits_flat.shape[-1]

        # Reshape
        router_logits = router_logits_flat.view(batch_size, seq_len, num_experts)

        # Extract only response tokens
        response_router_logits = router_logits[0, response_start:response_end, :]

        # Get top expert for each token
        top_experts = torch.argmax(response_router_logits, dim=-1).cpu().numpy()

        # Get top-k experts and probabilities
        routing_probs = torch.nn.functional.softmax(response_router_logits, dim=-1)
        top_k_probs, top_k_experts = torch.topk(
            routing_probs,
            k=min(4, num_experts),
            dim=-1
        )

        layer_routing[f"layer_{layer_idx}"] = {
            "top_expert": top_experts.tolist(),
            "top_k_experts": top_k_experts.cpu().numpy().tolist(),
            "top_k_probs": top_k_probs.float().cpu().numpy().tolist(),
            "num_experts": num_experts
        }

    # Get response tokens for reference
    response_token_ids = full_ids[0, response_start:response_end].cpu().tolist()
    response_tokens = [
        model_base.tokenizer.decode([tok]) for tok in response_token_ids
    ]

    return {
        'prompt': prompt,
        'response': response,
        'label': label,
        'prompt_length': prompt_len,
        'response_length': response_len,
        'response_start_pos': response_start,
        'response_end_pos': response_end,
        'response_tokens': response_tokens,
        'response_token_ids': response_token_ids,
        'layer_routing': layer_routing
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract expert routing during response generation using BeaverTails"
    )
    parser.add_argument(
        '--n_pairs',
        type=int,
        default=100,
        help='Number of matched pairs to analyze'
    )
    parser.add_argument(
        '--max_response_tokens',
        type=int,
        default=50,
        help='Maximum number of response tokens to analyze'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/media/volume/align_2_stg/alignment/expert_routing_analysis',
        help='Output directory for results'
    )
    parser.add_argument(
        '--beavertails_split',
        type=str,
        default='30k_test',
        help='BeaverTails split to use (30k_train, 30k_test, etc.)'
    )
    parser.add_argument(
        '--batch_process',
        action='store_true',
        help='Process in batches (experimental)'
    )

    args = parser.parse_args()

    # Configuration
    model_path = "openai/gpt-oss-20b"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load BeaverTails matched pairs
    print("="*80)
    print("BEAVERTAILS EXPERT ROUTING EXTRACTION")
    print("Routing During Response Generation (Refusal vs. Harmful)")
    print("="*80)

    matched_pairs = load_beavertails_matched_pairs(
        n_pairs=args.n_pairs,
        split=args.beavertails_split
    )

    print(f"\nAnalyzing {len(matched_pairs)} matched pairs")
    print(f"Max response tokens: {args.max_response_tokens}")

    # Load model
    print(f"\nLoading model: {model_path}")
    model_base = construct_model_base(model_path)

    # Get model info
    num_layers = len(model_base.model.model.layers)
    num_experts = model_base.model.config.num_local_experts
    experts_per_token = model_base.model.config.experts_per_token

    print(f"\nModel info:")
    print(f"  Layers: {num_layers}")
    print(f"  Experts per layer: {num_experts}")
    print(f"  Active experts per token: {experts_per_token}")

    # Extract routing for refusal responses
    print("\n" + "="*80)
    print("EXTRACTING ROUTING FOR REFUSAL RESPONSES")
    print("="*80)

    refusal_results = []
    for pair in tqdm(matched_pairs, desc="Processing refusal responses"):
        result = extract_routing_for_response(
            model_base=model_base,
            prompt=pair['prompt'],
            response=pair['refusal_response'],
            label='refusal',
            max_response_tokens=args.max_response_tokens
        )
        if result is not None:
            refusal_results.append(result)

    # Extract routing for harmful responses
    print("\n" + "="*80)
    print("EXTRACTING ROUTING FOR HARMFUL RESPONSES")
    print("="*80)

    harmful_results = []
    for pair in tqdm(matched_pairs, desc="Processing harmful responses"):
        result = extract_routing_for_response(
            model_base=model_base,
            prompt=pair['prompt'],
            response=pair['harmful_response'],
            label='harmful',
            max_response_tokens=args.max_response_tokens
        )
        if result is not None:
            harmful_results.append(result)

    # Combine results
    all_results = {
        "model": model_path,
        "dataset": "BeaverTails",
        "beavertails_split": args.beavertails_split,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "experts_per_token": experts_per_token,
        "n_pairs": len(matched_pairs),
        "max_response_tokens": args.max_response_tokens,
        "num_refusal": len(refusal_results),
        "num_harmful": len(harmful_results),
        "refusal_results": refusal_results,
        "harmful_results": harmful_results,
        "analysis_type": "response_tokens"
    }

    # Save results
    output_file = output_dir / "beavertails_routing_data.json"
    print(f"\n" + "="*80)
    print(f"Saving results to {output_file}")
    print("="*80)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\nSUMMARY")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Dataset: BeaverTails ({args.beavertails_split})")
    print(f"Matched pairs processed: {len(matched_pairs)}")
    print(f"Refusal responses analyzed: {len(refusal_results)}")
    print(f"Harmful responses analyzed: {len(harmful_results)}")
    print(f"Max response tokens: {args.max_response_tokens}")
    print(f"\nResults saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    print("\nNext steps:")
    print("  1. Analyze routing patterns during refusal vs harmful responses")
    print("  2. Compare expert selection across response tokens")
    print("  3. Identify which experts are associated with refusal behavior")
    print("  4. Look for token-position-specific patterns in routing")
    print("="*80)


if __name__ == "__main__":
    main()
