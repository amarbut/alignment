"""
Extract expert routing patterns for harmful vs harmless prompts from unsloth checkpoints.

This is an adapted version of extract_expert_routing_v2.py that:
1. Works with unsloth models and LoRA adapters
2. Handles PEFT-wrapped model structure
3. Uses the same harmful/harmless datasets from the Arditi pipeline
4. Analyzes the last 5 tokens of each prompt
5. Can process multiple checkpoints for comparison

Usage:
    # Single checkpoint
    python extract_expert_routing_unsloth.py \
        --checkpoint_dir checkpoints/expert_dropout_0.0/final

    # Multiple checkpoints
    python extract_expert_routing_unsloth.py \
        --batch_checkpoints \
        --checkpoint_dirs checkpoints/expert_dropout_0.0/final \
                         checkpoints/expert_dropout_0.3/final
"""
import random
import os
import sys
import torch
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
import numpy as np

# Add alignment directory to path
alignment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if alignment_dir not in sys.path:
    sys.path.insert(0, alignment_dir)

#import fix_hf_cache
from unsloth import FastLanguageModel
from peft import PeftModel

# Global storage for router outputs
router_outputs = {}


def create_mlp_hook(layer_idx):
    """Create a hook to capture router logits from MLP output for a specific layer."""
    def hook(module, input, output):
        # MLP output is a tuple: (hidden_states, router_logits)
        # router_logits shape: (batch_size * seq_len, num_experts)
        if isinstance(output, tuple) and len(output) >= 2:
            router_logits = output[1]  # Second element is router logits
            router_outputs[layer_idx] = router_logits.detach().cpu()
    return hook


def load_dataset(path):
    """Load a dataset from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


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

        # Load base model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        # Load PEFT adapter
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


def extract_expert_routing(
    model,
    tokenizer,
    prompts: List[str],
    label: str,
    batch_size: int = 4,
    last_n_tokens: int = 5
) -> List[Dict]:
    """
    Extract expert routing information for a set of prompts.

    Args:
        model: The model (potentially PEFT-wrapped)
        tokenizer: The tokenizer
        prompts: List of prompt strings
        label: Label for this dataset ('harmful' or 'harmless')
        batch_size: Batch size for processing
        last_n_tokens: Number of tokens from the end to analyze

    Returns:
        List of dictionaries containing routing information
    """
    global router_outputs
    results = []

    # Get base model layers (navigate PEFT wrapping)
    base_model = get_base_model_layers(model)

    # Register hooks on all MLPs to capture router logits
    hooks = []
    for layer_idx, layer in enumerate(base_model.layers):
        hook = layer.mlp.register_forward_hook(create_mlp_hook(layer_idx))
        hooks.append(hook)

    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Processing {label} prompts"):
            batch_prompts = prompts[i:i + batch_size]

            # Format and tokenize prompts
            formatted_prompts = [format_prompt_for_model(tokenizer, p) for p in batch_prompts]

            tokenized = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            input_ids = tokenized.input_ids.to(model.device)
            attention_mask = tokenized.attention_mask.to(model.device)

            # Clear previous router outputs
            router_outputs = {}

            # Forward pass (hooks will capture router outputs)
            try:
                with torch.no_grad():
                    _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
            except (RuntimeError, TypeError) as e:
                # Model may throw errors, but hooks still capture data
                if len(router_outputs) == 0:
                    # If no data was captured, re-raise the error
                    raise
                # Otherwise continue - we got the router logits we need

            # Process each sample in the batch
            for batch_idx in range(len(batch_prompts)):
                # Get the sequence length for this sample (accounting for padding)
                seq_len = attention_mask[batch_idx].sum().item()

                # Focus on the last N tokens
                start_pos = max(0, seq_len - last_n_tokens)
                end_pos = seq_len

                # Extract routing info for each MoE layer
                layer_routing = {}
                for layer_idx in sorted(router_outputs.keys()):
                    # router_logits shape from MLP: (batch_size * seq_len, num_experts)
                    # Need to reshape to (batch_size, seq_len, num_experts)
                    router_logits_flat = router_outputs[layer_idx]
                    batch_size_cur = len(batch_prompts)
                    seq_len_padded = input_ids.shape[1]
                    num_experts = router_logits_flat.shape[-1]

                    router_logits = router_logits_flat.view(batch_size_cur, seq_len_padded, num_experts)
                    layer_router = router_logits[batch_idx, start_pos:end_pos, :]

                    # Get top expert indices for each token position
                    top_experts = torch.argmax(layer_router, dim=-1).cpu().numpy()

                    # Get top-4 experts (since the model uses top-k routing)
                    routing_probs = torch.nn.functional.softmax(layer_router, dim=-1)
                    top_k_probs, top_k_experts = torch.topk(
                        routing_probs,
                        k=min(4, layer_router.shape[-1]),
                        dim=-1
                    )

                    # Store full routing distribution for later analysis
                    layer_routing[f"layer_{layer_idx}"] = {
                        "top_expert": top_experts.tolist(),  # Single top expert per token
                        "top_k_experts": top_k_experts.cpu().numpy().tolist(),  # Top-4 experts per token
                        "top_k_probs": top_k_probs.float().cpu().numpy().tolist(),  # Their probabilities
                        "num_experts": num_experts
                    }

                results.append({
                    "prompt": batch_prompts[batch_idx],
                    "label": label,
                    "prompt_length": seq_len,
                    "analyzed_positions": list(range(start_pos, end_pos)),
                    "layer_routing": layer_routing
                })

    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    return results


def process_checkpoint(
    checkpoint_dir: Optional[str],
    checkpoint_name: str,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    batch_size: int = 4,
    last_n_tokens: int = 5,
    base_model_name: str = "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    max_seq_length: int = 512
) -> Dict:
    """
    Process a single checkpoint and extract routing for harmful/harmless prompts.

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
        num_experts = 32  # Default for GPT-OSS-20B

    if hasattr(model.config, 'experts_per_token'):
        experts_per_token = model.config.experts_per_token
    else:
        experts_per_token = 2  # Default

    print(f"\nModel info:")
    print(f"  Layers: {num_layers}")
    print(f"  Experts per layer: {num_experts}")
    print(f"  Active experts per token: {experts_per_token}")
    print(f"  Has LoRA adapter: {is_adapted}")

    # Extract routing for harmful prompts
    print("\n" + "="*80)
    print("EXTRACTING ROUTING FOR HARMFUL PROMPTS")
    print("="*80)

    harmful_results = extract_expert_routing(
        model=model,
        tokenizer=tokenizer,
        prompts=harmful_prompts,
        label="harmful",
        batch_size=batch_size,
        last_n_tokens=last_n_tokens
    )

    # Extract routing for harmless prompts
    print("\n" + "="*80)
    print("EXTRACTING ROUTING FOR HARMLESS PROMPTS")
    print("="*80)

    harmless_results = extract_expert_routing(
        model=model,
        tokenizer=tokenizer,
        prompts=harmless_prompts,
        label="harmless",
        batch_size=batch_size,
        last_n_tokens=last_n_tokens
    )

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
        "last_n_tokens": last_n_tokens,
        "num_harmful": len(harmful_results),
        "num_harmless": len(harmless_results),
        "harmful_results": harmful_results,
        "harmless_results": harmless_results
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract expert routing from unsloth checkpoints using harmful/harmless datasets"
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
    print("EXPERT ROUTING EXTRACTION (UNSLOTH)")
    print("Harmful vs. Harmless Prompt Routing Analysis")
    print("="*80)

    # Load datasets
    print("\nLoading datasets...")
    harmful_data = load_dataset(args.harmful_dataset)
    harmless_data = load_dataset(args.harmless_dataset)

    harmful_prompts = [item["instruction"] for item in harmful_data]
    harmless_prompts = [item["instruction"] for item in random.sample(harmless_data, 200)]

    print(f"Loaded {len(harmful_prompts)} harmful prompts")
    print(f"Loaded {len(harmless_prompts)} harmless prompts")
    print(f"Analyzing last {args.last_n_tokens} tokens")

    # Determine checkpoints to process
    if args.batch_checkpoints:
        if args.checkpoint_dirs is None:
            raise ValueError("--checkpoint_dirs required when using --batch_checkpoints")
        checkpoints = []
        for d in args.checkpoint_dirs:
            # Include parent directory in name to avoid collisions (e.g., "dropout_0.0_final")
            path = Path(d)
            parent_name = path.parent.name
            checkpoint_name = f"{parent_name}_{path.name}"
            checkpoints.append((checkpoint_name, d))
    else:
        if args.checkpoint_dir:
            # Include parent directory for single checkpoint too
            path = Path(args.checkpoint_dir)
            checkpoint_name = f"{path.parent.name}_{path.name}"
        else:
            checkpoint_name = "base_model"
        checkpoints = [(checkpoint_name, args.checkpoint_dir)]

    print(f"\nProcessing {len(checkpoints)} checkpoint(s)")

    # Process each checkpoint
    all_checkpoint_results = []

    for checkpoint_name, checkpoint_dir in checkpoints:
        result = process_checkpoint(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            harmful_prompts=harmful_prompts,
            harmless_prompts=harmless_prompts,
            batch_size=args.batch_size,
            last_n_tokens=args.last_n_tokens,
            base_model_name=args.base_model,
            max_seq_length=args.max_seq_length
        )
        all_checkpoint_results.append(result)

    # Save results
    if args.batch_checkpoints:
        output_file = output_dir / "expert_routing_comparison.json"
    else:
        checkpoint_name = checkpoints[0][0]
        output_file = output_dir / f"expert_routing_{checkpoint_name}.json"

    print(f"\n" + "="*80)
    print(f"Saving results to {output_file}")
    print("="*80)

    results = {
        "dataset_harmful": args.harmful_dataset,
        "dataset_harmless": args.harmless_dataset,
        "base_model": args.base_model,
        "num_checkpoints": len(checkpoints),
        "checkpoints": all_checkpoint_results
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\nSUMMARY")
    print("="*80)
    print(f"Base model: {args.base_model}")
    print(f"Harmful dataset: {args.harmful_dataset}")
    print(f"Harmless dataset: {args.harmless_dataset}")
    print(f"Checkpoints processed: {len(checkpoints)}")

    for result in all_checkpoint_results:
        print(f"\n  {result['checkpoint_name']}:")
        print(f"    Has adapter: {result['has_adapter']}")
        print(f"    Harmful prompts: {result['num_harmful']}")
        print(f"    Harmless prompts: {result['num_harmless']}")
        print(f"    Last N tokens: {result['last_n_tokens']}")

    print(f"\nResults saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / (1024*1024):.2f} MB")

    print("\nNext steps:")
    print("  1. Analyze routing pattern differences between harmful and harmless")
    print("  2. Compare how dropout affected expert specialization")
    print("  3. Identify experts that became more/less active after fine-tuning")
    print("  4. Measure routing entropy and diversity changes")
    print("="*80)


if __name__ == "__main__":
    main()
