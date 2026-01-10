"""
Extract expert routing patterns from fine-tuned unsloth checkpoints using BeaverTails dataset.

This is a modified version of extract_beavertails_routing.py that:
1. Loads unsloth models with 4-bit quantization
2. Supports loading LoRA adapters from fine-tuning checkpoints
3. Handles PEFT-wrapped model structure
4. Can process multiple checkpoints in batch for comparison

Usage:
    # Single checkpoint
    python extract_beavertails_routing_unsloth.py \
        --checkpoint_dir checkpoints/expert_dropout_0.0/final \
        --n_pairs 100

    # Multiple checkpoints (batch mode)
    python extract_beavertails_routing_unsloth.py \
        --batch_checkpoints \
        --checkpoint_dirs checkpoints/expert_dropout_0.0/final \
                         checkpoints/expert_dropout_0.3/final \
                         checkpoints/expert_dropout_0.5/checkpoint-500 \
        --n_pairs 100
"""

import os
import sys
import torch
import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import Counter

# Add alignment directory to path
alignment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if alignment_dir not in sys.path:
    sys.path.insert(0, alignment_dir)

import fix_hf_cache
from unsloth import FastLanguageModel

# Global storage for router outputs
router_outputs = {}


def create_mlp_hook(layer_idx):
    """
    Create a hook to capture router logits from MLP output.

    The MLP output is a tuple: (hidden_states, router_logits).
    Works with both base model and PEFT-wrapped models.
    """
    def hook(module, input, output):
        # MLP output is a tuple: (hidden_states, router_logits)
        # router_logits shape: (batch_size * seq_len, num_experts)
        if isinstance(output, tuple) and len(output) >= 2:
            router_logits = output[1]  # Second element is router logits
            router_outputs[layer_idx] = router_logits.detach().cpu()
    return hook


# Common refusal templates (variety of phrasings)
REFUSAL_TEMPLATES = [
    "I'm sorry, but I can't help with that.",
    "I cannot assist with that request.",
    "I'm not able to help with requests like this.",
    "I can't help with that.",
    "I'm sorry, I cannot assist with that.",
    "I'm unable to help with this request.",
    "I can't provide assistance with that.",
    "I'm afraid I can't help with that.",
    "I'm not able to assist with that.",
    "I cannot help with that request.",
    "Sorry, but I can't help with that.",
    "I'm unable to provide that assistance.",
    "I can't help you with that.",
    "I'm not comfortable helping with that.",
    "I cannot provide help with that request.",
]


def load_beavertails_matched_pairs(
    n_pairs: int = 100,
    split: str = '30k_test',
    refusal_template: str = None,
    randomize_refusals: bool = True,
    seed: int = 42
):
    """
    Load BeaverTails dataset and create matched pairs with synthetic refusals.

    Returns:
        List of matched pairs, each containing:
        {
            'prompt': str,
            'refusal_response': str (synthetic),
            'harmful_response': str (from BeaverTails)
        }
    """
    print(f"Loading BeaverTails dataset (split: {split})...")

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Load BeaverTails from HuggingFace
    dataset = load_dataset("PKU-Alignment/BeaverTails", split=split)

    print(f"Loaded {len(dataset)} examples")

    # Filter to unsafe examples (these have harmful responses)
    unsafe_examples = [item for item in dataset if not item['is_safe']]

    print(f"Found {len(unsafe_examples)} unsafe examples")

    # Create matched pairs with synthetic refusals
    matched_pairs = []
    refusal_counts = Counter()

    for item in unsafe_examples[:n_pairs]:
        # Select refusal template
        if refusal_template is not None:
            refusal_text = refusal_template
        elif randomize_refusals:
            refusal_text = np.random.choice(REFUSAL_TEMPLATES)
            refusal_counts[refusal_text] += 1
        else:
            refusal_text = REFUSAL_TEMPLATES[0]

        matched_pairs.append({
            'prompt': item['prompt'],
            'refusal_response': refusal_text,
            'harmful_response': item['response']
        })

    print(f"\nCreated {len(matched_pairs)} matched pairs with synthetic refusals")

    if refusal_template is not None:
        print(f"Using fixed template: '{refusal_template}'")
    elif randomize_refusals:
        print(f"Using {len(REFUSAL_TEMPLATES)} randomized refusal templates")
    else:
        print(f"Using default template: '{REFUSAL_TEMPLATES[0]}'")

    return matched_pairs


def load_model_with_adapter(
    base_model_name: str = "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    checkpoint_dir: Optional[str] = None,
    max_seq_length: int = 512
):
    """
    Load unsloth model with optional LoRA adapter from checkpoint.

    Args:
        base_model_name: Base model to load
        checkpoint_dir: Path to checkpoint with LoRA adapter (optional)
        max_seq_length: Maximum sequence length

    Returns:
        (model, tokenizer, is_adapted)
    """
    print(f"\nLoading model: {base_model_name}")

    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")

        print(f"Loading LoRA adapter from: {checkpoint_dir}")

        # Load with adapter - unsloth will detect and load the adapter
        from peft import PeftModel

        # First load base model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        # Then load PEFT adapter
        model = PeftModel.from_pretrained(model, checkpoint_dir)

        print("✓ Loaded model with LoRA adapter")
        return model, tokenizer, True
    else:
        # Load base model only
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        print("✓ Loaded base model (no adapter)")
        return model, tokenizer, False


def get_base_model_layers(model):
    """
    Navigate PEFT wrapping to get to the actual model layers.

    PEFT structure: PeftModel -> base_model -> model -> model -> layers
    Base structure: model -> model -> layers
    """
    current = model

    # Navigate through wrapping
    while hasattr(current, 'model') and not hasattr(current, 'layers'):
        current = current.model

    if not hasattr(current, 'layers'):
        raise ValueError("Could not find model.layers - unexpected model structure")

    return current


def format_prompt_for_model(tokenizer, prompt: str) -> str:
    """
    Format prompt using the model's chat template.

    Args:
        tokenizer: The model tokenizer
        prompt: Raw prompt text

    Returns:
        Formatted prompt string
    """
    # Use chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted
    else:
        # Fallback: simple formatting
        return f"<|user|>\n{prompt}\n<|assistant|>\n"


def extract_routing_for_response(
    model,
    tokenizer,
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
        model: The model (potentially PEFT-wrapped)
        tokenizer: The tokenizer
        prompt: The input prompt
        response: The response to analyze
        label: 'refusal' or 'harmful'
        max_response_tokens: Maximum response tokens to analyze

    Returns:
        Dictionary with routing information for response tokens
    """
    global router_outputs

    # Format prompt
    formatted_prompt = format_prompt_for_model(tokenizer, prompt)

    # Tokenize prompt to get length
    prompt_tokenized = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=False,
        truncation=False
    )
    prompt_ids = prompt_tokenized.input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    # Tokenize full sequence (prompt + response)
    full_text = formatted_prompt + response
    full_tokenized = tokenizer(
        full_text,
        return_tensors="pt",
        padding=False,
        truncation=False
    )

    full_ids = full_tokenized.input_ids.to(model.device)
    attention_mask = torch.ones_like(full_ids)

    # Determine response token positions
    response_start = prompt_len
    response_end = min(full_ids.shape[1], prompt_len + max_response_tokens)
    response_len = response_end - response_start

    if response_len <= 0:
        return None

    # Get base model layers (navigate PEFT wrapping)
    base_model = get_base_model_layers(model)

    # Register hooks on all MLPs
    hooks = []
    for layer_idx, layer in enumerate(base_model.layers):
        hook = layer.mlp.register_forward_hook(create_mlp_hook(layer_idx))
        hooks.append(hook)

    router_outputs = {}

    try:
        # Forward pass through full sequence
        with torch.no_grad():
            _ = model(
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
        router_logits_flat = router_outputs[layer_idx]

        batch_size = 1
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
        tokenizer.decode([tok]) for tok in response_token_ids
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


def process_checkpoint(
    checkpoint_dir: Optional[str],
    checkpoint_name: str,
    matched_pairs: List[Dict],
    max_response_tokens: int,
    base_model_name: str = "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    max_seq_length: int = 512
) -> Dict:
    """
    Process a single checkpoint and extract routing for all matched pairs.

    Returns:
        Dictionary with all results for this checkpoint
    """
    print("\n" + "="*80)
    print(f"PROCESSING CHECKPOINT: {checkpoint_name}")
    print("="*80)

    # Load model
    model, tokenizer, is_adapted = load_model_with_adapter(
        base_model_name=base_model_name,
        checkpoint_dir=checkpoint_dir,
        max_seq_length=max_seq_length
    )

    # Get model info
    base_model = get_base_model_layers(model)
    num_layers = len(base_model.layers)

    # Get expert count from config
    if hasattr(model.config, 'num_local_experts'):
        num_experts = model.config.num_local_experts
    else:
        # Default for GPT-OSS-20B
        num_experts = 32

    if hasattr(model.config, 'experts_per_token'):
        experts_per_token = model.config.experts_per_token
    else:
        experts_per_token = 2  # Default

    print(f"\nModel info:")
    print(f"  Layers: {num_layers}")
    print(f"  Experts per layer: {num_experts}")
    print(f"  Active experts per token: {experts_per_token}")
    print(f"  Has LoRA adapter: {is_adapted}")

    # Extract routing for refusal responses
    print("\n" + "="*80)
    print("EXTRACTING ROUTING FOR REFUSAL RESPONSES")
    print("="*80)

    refusal_results = []
    for pair in tqdm(matched_pairs, desc="Processing refusal responses"):
        result = extract_routing_for_response(
            model=model,
            tokenizer=tokenizer,
            prompt=pair['prompt'],
            response=pair['refusal_response'],
            label='refusal',
            max_response_tokens=max_response_tokens
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
            model=model,
            tokenizer=tokenizer,
            prompt=pair['prompt'],
            response=pair['harmful_response'],
            label='harmful',
            max_response_tokens=max_response_tokens
        )
        if result is not None:
            harmful_results.append(result)

    # Clean up
    del model
    torch.cuda.empty_cache()

    return {
        "checkpoint_name": checkpoint_name,
        "checkpoint_dir": checkpoint_dir,
        "base_model": base_model_name,
        "has_adapter": is_adapted,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "experts_per_token": experts_per_token,
        "n_pairs": len(matched_pairs),
        "max_response_tokens": max_response_tokens,
        "num_refusal": len(refusal_results),
        "num_harmful": len(harmful_results),
        "refusal_results": refusal_results,
        "harmful_results": harmful_results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract expert routing from unsloth checkpoints using BeaverTails"
    )

    # Checkpoint arguments
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help='Path to checkpoint directory with LoRA adapter (None for base model)'
    )
    parser.add_argument(
        '--batch_checkpoints',
        action='store_true',
        help='Process multiple checkpoints (use with --checkpoint_dirs)'
    )
    parser.add_argument(
        '--checkpoint_dirs',
        nargs='+',
        default=None,
        help='Multiple checkpoint directories to process in batch'
    )

    # Model arguments
    parser.add_argument(
        '--base_model',
        type=str,
        default='unsloth/gpt-oss-20b-unsloth-bnb-4bit',
        help='Base model to load'
    )
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )

    # Data arguments
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
        '--beavertails_split',
        type=str,
        default='30k_test',
        help='BeaverTails split to use'
    )
    parser.add_argument(
        '--refusal_template',
        type=str,
        default=None,
        help='Specific refusal template (overrides randomization)'
    )
    parser.add_argument(
        '--no_randomize',
        action='store_true',
        help='Disable randomization and use fixed template'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for refusal selection'
    )

    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='expert_routing_analysis',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*80)
    print("BEAVERTAILS EXPERT ROUTING EXTRACTION (UNSLOTH)")
    print("Routing During Response Generation (Refusal vs. Harmful)")
    print("="*80)

    # Load BeaverTails matched pairs (shared across all checkpoints)
    matched_pairs = load_beavertails_matched_pairs(
        n_pairs=args.n_pairs,
        split=args.beavertails_split,
        refusal_template=args.refusal_template,
        randomize_refusals=not args.no_randomize,
        seed=args.seed
    )

    print(f"\nAnalyzing {len(matched_pairs)} matched pairs")
    print(f"Max response tokens: {args.max_response_tokens}")

    # Determine checkpoints to process
    if args.batch_checkpoints:
        if args.checkpoint_dirs is None:
            raise ValueError("--checkpoint_dirs required when using --batch_checkpoints")
        checkpoints = [
            (Path(d).name, d) for d in args.checkpoint_dirs
        ]
    else:
        checkpoint_name = Path(args.checkpoint_dir).name if args.checkpoint_dir else "base_model"
        checkpoints = [(checkpoint_name, args.checkpoint_dir)]

    print(f"\nProcessing {len(checkpoints)} checkpoint(s)")

    # Process each checkpoint
    all_checkpoint_results = []

    for checkpoint_name, checkpoint_dir in checkpoints:
        result = process_checkpoint(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            matched_pairs=matched_pairs,
            max_response_tokens=args.max_response_tokens,
            base_model_name=args.base_model,
            max_seq_length=args.max_seq_length
        )
        all_checkpoint_results.append(result)

    # Save results
    if args.batch_checkpoints:
        output_file = output_dir / "beavertails_routing_comparison.json"
    else:
        checkpoint_name = checkpoints[0][0]
        output_file = output_dir / f"beavertails_routing_{checkpoint_name}.json"

    print(f"\n" + "="*80)
    print(f"Saving results to {output_file}")
    print("="*80)

    results = {
        "dataset": "BeaverTails",
        "beavertails_split": args.beavertails_split,
        "base_model": args.base_model,
        "num_checkpoints": len(checkpoints),
        "checkpoints": all_checkpoint_results,
        "analysis_type": "response_tokens"
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\nSUMMARY")
    print("="*80)
    print(f"Base model: {args.base_model}")
    print(f"Dataset: BeaverTails ({args.beavertails_split})")
    print(f"Checkpoints processed: {len(checkpoints)}")

    for result in all_checkpoint_results:
        print(f"\n  {result['checkpoint_name']}:")
        print(f"    Has adapter: {result['has_adapter']}")
        print(f"    Refusal responses: {result['num_refusal']}")
        print(f"    Harmful responses: {result['num_harmful']}")

    print(f"\nResults saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / (1024*1024):.2f} MB")

    print("\nNext steps:")
    print("  1. Analyze routing pattern differences between base and fine-tuned models")
    print("  2. Compare how dropout affected expert selection")
    print("  3. Identify which experts changed behavior after fine-tuning")
    print("  4. Visualize routing differences for refusal vs harmful responses")
    print("="*80)


if __name__ == "__main__":
    main()
