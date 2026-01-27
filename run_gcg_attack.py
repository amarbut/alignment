#!/usr/bin/env python3
"""
Run GCG (Greedy Coordinate Gradient) attack on the OSS model.

GCG is a white-box attack that optimizes adversarial suffixes using gradient
information. This script handles the full attack pipeline including suffix
optimization, completion generation, and evaluation.

Usage:
    # Run on a few prompts for testing
    python run_gcg_attack.py --n_prompts 5 --num_steps 100

    # Full run with default settings
    python run_gcg_attack.py --n_prompts 50 --num_steps 250

    # Memory-efficient settings for tight GPU memory
    python run_gcg_attack.py --search_width 128 --batch_size 32

Requirements:
    pip install nanogcg  # Recommended for best performance

References:
    Paper: https://arxiv.org/abs/2307.15043
    GitHub: https://github.com/llm-attacks/llm-attacks
    nanogcg: https://github.com/GraySwanAI/nanoGCG
"""

import argparse
import gc
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm

# Local imports
from dataset.load_dataset import load_dataset_split, load_dataset
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from jailbreak_baselines import GCGAttack, GCGConfig, run_gcg_attack


def load_model(model_path: str, use_unsloth: bool = True):
    """
    Load the OSS model.

    For GCG, we need gradient access, so we load with gradient tracking enabled.
    """
    if use_unsloth:
        from unsloth import FastLanguageModel

        print(f"Loading model with unsloth: {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model with transformers: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # For GCG we need the model in training mode for gradient computation
    # but we'll toggle this in the attack itself
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def format_prompt_oss(instruction: str, system: str = None) -> str:
    """Format instruction using OSS chat template."""
    if system:
        return f"<|start|>developer<|message|>{system}<|end|><|start|>user<|message|>{instruction}<|end|><|start|>assistant<|channel|>final<|message|>"
    else:
        return f"<|start|>user<|message|>{instruction}<|end|><|start|>assistant<|channel|>final<|message|>"


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Generate a completion for a single prompt."""
    device = next(model.parameters()).device

    formatted = format_prompt_oss(prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        if temperature == 0:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )

    completion = tokenizer.decode(
        outputs[0][inputs["input_ids"].size(1):],
        skip_special_tokens=True,
    ).strip()

    return completion


def run_gcg_baseline(
    model,
    tokenizer,
    prompts: List[Dict[str, str]],
    output_dir: str,
    gcg_config: GCGConfig,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    eval_methodologies: List[str] = None,
    use_nanogcg: bool = False,
    cache_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run GCG attack on prompts and evaluate.

    Args:
        model: Target language model
        tokenizer: Model tokenizer
        prompts: List of prompt dicts
        output_dir: Output directory
        gcg_config: GCG configuration
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        eval_methodologies: Evaluation methods
        use_nanogcg: Use nanogcg library
        cache_path: Path to load/save suffix cache

    Returns:
        Results dictionary
    """
    if eval_methodologies is None:
        eval_methodologies = ["substring_matching", "openai"]

    os.makedirs(output_dir, exist_ok=True)

    # Initialize attack
    attack = GCGAttack(
        model=model,
        tokenizer=tokenizer,
        config=gcg_config,
        use_nanogcg=use_nanogcg,
    )

    # Load cache if available
    if cache_path and os.path.exists(cache_path):
        print(f"Loading suffix cache from {cache_path}")
        attack.load_cache(cache_path)

    results = {
        "attack": "GCG",
        "config": {
            "num_steps": gcg_config.num_steps,
            "search_width": gcg_config.search_width,
            "topk": gcg_config.topk,
            "batch_size": gcg_config.batch_size,
            "use_nanogcg": use_nanogcg,
        },
        "completions": [],
        "optimization_history": [],
    }

    for i, prompt_dict in enumerate(tqdm(prompts, desc="GCG Attack")):
        original_prompt = prompt_dict["instruction"]
        category = prompt_dict.get("category", "unknown")

        print(f"\n[{i+1}/{len(prompts)}] Processing: {original_prompt[:50]}...")

        # Check if already cached
        if original_prompt in attack.suffix_cache:
            suffix = attack.suffix_cache[original_prompt]
            loss = 0.0  # Unknown for cached
            history = []
            print(f"  Using cached suffix")
        else:
            # Optimize suffix
            print(f"  Optimizing suffix ({gcg_config.num_steps} steps)...")
            suffix, loss, history = attack.optimize_suffix(original_prompt)
            attack.suffix_cache[original_prompt] = suffix
            print(f"  Final loss: {loss:.4f}")

        # Create adversarial prompt
        adversarial_prompt = f"{original_prompt} {suffix}"

        # Generate completion
        print(f"  Generating completion...")
        completion = generate_completion(
            model, tokenizer, adversarial_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        results["completions"].append({
            "prompt": original_prompt,
            "transformed_prompt": adversarial_prompt,
            "suffix": suffix,
            "response": completion,
            "category": category,
            "final_loss": loss,
        })

        results["optimization_history"].append({
            "prompt_idx": i,
            "final_loss": loss,
            "history_summary": history[-5:] if history else [],
        })

        # Save intermediate results
        if (i + 1) % 3 == 0:
            intermediate_path = os.path.join(output_dir, "gcg_intermediate.json")
            with open(intermediate_path, "w") as f:
                json.dump(results, f, indent=2)

            # Save cache
            cache_save_path = os.path.join(output_dir, "suffix_cache.json")
            attack.save_cache(cache_save_path)

        gc.collect()
        torch.cuda.empty_cache()

    # Save final cache
    cache_save_path = os.path.join(output_dir, "suffix_cache.json")
    attack.save_cache(cache_save_path)
    print(f"Saved suffix cache to {cache_save_path}")

    # Save raw completions
    completions_path = os.path.join(output_dir, "gcg_completions.json")
    with open(completions_path, "w") as f:
        json.dump(results["completions"], f, indent=2)
    print(f"Saved completions to {completions_path}")

    # Format for evaluation (use original prompt for judging)
    eval_completions = [
        {
            "prompt": c["prompt"],
            "response": c["response"],
            "category": c["category"],
        }
        for c in results["completions"]
    ]

    # Evaluate
    print("\nEvaluating results...")
    eval_path = os.path.join(output_dir, "gcg_evaluation.json")
    evaluation = evaluate_jailbreak(
        completions=eval_completions,
        methodologies=eval_methodologies,
        evaluation_path=eval_path,
    )

    # Merge evaluation into results
    results["evaluation"] = {
        k: v for k, v in evaluation.items()
        if k != "completions"
    }

    # Save final results
    final_path = os.path.join(output_dir, "gcg_results.json")
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved final results to {final_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run GCG attack on OSS model"
    )

    # Model settings
    parser.add_argument(
        "--model_path",
        type=str,
        default="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        help="Model path or HF model ID",
    )
    parser.add_argument(
        "--use_unsloth",
        action="store_true",
        default=True,
        help="Use unsloth for model loading",
    )
    parser.add_argument(
        "--no_unsloth",
        action="store_true",
        help="Don't use unsloth",
    )

    # Dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="harmful_test",
        help="Dataset to use",
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=None,
        help="Number of prompts (default: all)",
    )

    # GCG settings
    parser.add_argument(
        "--num_steps",
        type=int,
        default=250,
        help="Number of GCG optimization steps",
    )
    parser.add_argument(
        "--search_width",
        type=int,
        default=256,
        help="Number of candidates per step",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=128,
        help="Top-k tokens to consider",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for candidate evaluation",
    )
    parser.add_argument(
        "--use_nanogcg",
        action="store_true",
        default=False,
        help="Use nanogcg library if installed (not recommended due to dependency conflicts)",
    )
    parser.add_argument(
        "--suffix_length",
        type=int,
        default=20,
        help="Number of tokens in adversarial suffix (default: 20)",
    )

    # Generation settings
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Max tokens to generate for evaluation (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )

    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pipeline/runs/gcg_attack",
        help="Output directory",
    )
    parser.add_argument(
        "--eval_methodologies",
        type=str,
        nargs="+",
        default=["substring_matching", "openai"],
        help="Evaluation methodologies",
    )

    # Cache settings
    parser.add_argument(
        "--cache_path",
        type=str,
        default=None,
        help="Path to existing suffix cache",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Handle flags
    use_unsloth = args.use_unsloth and not args.no_unsloth
    use_nanogcg = args.use_nanogcg

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset in ["harmful_test", "harmful_val", "harmful_train"]:
        split = args.dataset.replace("harmful_", "")
        prompts = load_dataset_split("harmful", split)
    else:
        prompts = load_dataset(args.dataset)

    if args.n_prompts:
        prompts = prompts[:args.n_prompts]

    print(f"Loaded {len(prompts)} prompts")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_dict = vars(args).copy()
    config_dict["timestamp"] = timestamp
    config_dict["n_prompts_actual"] = len(prompts)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Create GCG config
    # Generate initial suffix string based on suffix_length
    optim_str_init = " ".join(["!"] * args.suffix_length)
    print(f"Suffix length: {args.suffix_length} tokens")

    gcg_config = GCGConfig(
        num_steps=args.num_steps,
        search_width=args.search_width,
        topk=args.topk,
        batch_size=args.batch_size,
        seed=args.seed,
        optim_str_init=optim_str_init,
    )

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(args.model_path, use_unsloth=use_unsloth)

    # Check for nanogcg
    if use_nanogcg:
        try:
            import nanogcg
            print(f"nanogcg version: {nanogcg.__version__ if hasattr(nanogcg, '__version__') else 'unknown'}")
        except ImportError:
            print("WARNING: nanogcg not installed. Install with: pip install nanogcg")
            print("Falling back to custom implementation.")
            use_nanogcg = False

    # Run attack
    print("\nStarting GCG attack...")
    results = run_gcg_baseline(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        output_dir=output_dir,
        gcg_config=gcg_config,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        eval_methodologies=args.eval_methodologies,
        use_nanogcg=use_nanogcg,
        cache_path=args.cache_path,
    )

    # Print summary
    print("\n" + "="*60)
    print("GCG Attack Complete")
    print("="*60)
    print(f"Results saved to: {output_dir}")

    if "evaluation" in results:
        for key, value in results["evaluation"].items():
            if "success_rate" in key:
                print(f"{key}: {value:.4f}")

    print("\nTo run evaluation on more methods, use:")
    print(f"  python -c \"from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak; ...")


if __name__ == "__main__":
    main()
