#!/usr/bin/env python3
"""
Run black-box jailbreak baseline attacks on the OSS model.

This script runs ArtPrompt, DeepInception, and FFA attacks on harmful prompts
and evaluates them using the existing jailbreak evaluation framework.

Usage:
    python run_jailbreak_baselines.py --method artprompt --n_prompts 50
    python run_jailbreak_baselines.py --method all --output_dir results/baselines
"""

import argparse
import json
import os
import gc
from datetime import datetime
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm

# Local imports
from dataset.load_dataset import load_dataset_split, load_dataset
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from jailbreak_baselines import (
    ArtPromptAttack,
    DeepInceptionAttack,
    FFAAttack,
    JailbreakMethod,
)


def load_model(model_path: str, use_unsloth: bool = True):
    """
    Load the OSS model.

    Args:
        model_path: Path or HF model ID
        use_unsloth: Whether to use unsloth for 4-bit loading

    Returns:
        model, tokenizer tuple
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


def generate_completions(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 4,
    system_prompt: Optional[str] = None,
) -> List[str]:
    """
    Generate completions for a list of prompts.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompts to complete
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 for greedy)
        batch_size: Batch size for generation
        system_prompt: Optional system prompt

    Returns:
        List of generated completions
    """
    completions = []
    device = next(model.parameters()).device

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i + batch_size]

        # Format prompts with chat template
        formatted_prompts = [
            format_prompt_oss(p, system=system_prompt) for p in batch_prompts
        ]

        # Tokenize
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048 - max_new_tokens,
        ).to(device)

        # Generate
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

        # Decode, removing the input prompt
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True,
            )
            completions.append(generated.strip())

        # Clear cache periodically
        if i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()

    return completions


def get_attack_method(method_name: str, **kwargs) -> JailbreakMethod:
    """Get attack method by name."""
    methods = {
        "artprompt": ArtPromptAttack,
        "deepinception": DeepInceptionAttack,
        "ffa": FFAAttack,
    }

    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(methods.keys())}")

    return methods[method_name](**kwargs)


def run_baseline_attack(
    model,
    tokenizer,
    attack: JailbreakMethod,
    prompts: List[Dict[str, str]],
    output_dir: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 4,
    system_prompt: Optional[str] = None,
    eval_methodologies: List[str] = None,
) -> Dict[str, Any]:
    """
    Run a single baseline attack and evaluate.

    Args:
        model: The language model
        tokenizer: The tokenizer
        attack: The jailbreak attack method
        prompts: List of prompt dicts with 'instruction' and 'category'
        output_dir: Directory to save results
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        batch_size: Batch size for generation
        system_prompt: Optional system prompt
        eval_methodologies: Evaluation methodologies to use

    Returns:
        Evaluation results dict
    """
    if eval_methodologies is None:
        eval_methodologies = ["substring_matching", "openai"]

    print(f"\n{'='*60}")
    print(f"Running attack: {attack.name}")
    print(f"{'='*60}")

    # Transform prompts
    print("Transforming prompts...")
    original_instructions = [p["instruction"] for p in prompts]
    transformed_prompts = attack.transform_batch(original_instructions)

    # Generate completions
    print("Generating completions...")
    completions = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=transformed_prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
        system_prompt=system_prompt,
    )

    # Format for evaluation
    # Note: We use the ORIGINAL prompt for evaluation, not the transformed one
    # This is because we want to check if the response fulfills the original harmful request
    completion_data = []
    for i, (prompt_dict, transformed, completion) in enumerate(
        zip(prompts, transformed_prompts, completions)
    ):
        completion_data.append({
            "prompt": prompt_dict["instruction"],  # Original for evaluation
            "transformed_prompt": transformed,  # For reference
            "response": completion,
            "category": prompt_dict.get("category", "unknown"),
        })

    # Save raw completions
    os.makedirs(output_dir, exist_ok=True)
    completions_path = os.path.join(output_dir, f"{attack.name}_completions.json")
    with open(completions_path, "w") as f:
        json.dump(completion_data, f, indent=2)
    print(f"Saved completions to {completions_path}")

    # Evaluate
    print("Evaluating...")
    eval_path = os.path.join(output_dir, f"{attack.name}_evaluation.json")
    results = evaluate_jailbreak(
        completions=completion_data,
        methodologies=eval_methodologies,
        evaluation_path=eval_path,
    )

    # Add attack metadata
    results["attack_metadata"] = attack.get_metadata()
    results["n_prompts"] = len(prompts)
    results["timestamp"] = datetime.now().isoformat()

    # Save updated results
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_all_baselines(
    model,
    tokenizer,
    prompts: List[Dict[str, str]],
    output_dir: str,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """Run all baseline attacks and return combined results."""
    methods = ["artprompt", "deepinception", "ffa"]
    all_results = {}

    for method_name in methods:
        attack = get_attack_method(method_name)
        results = run_baseline_attack(
            model=model,
            tokenizer=tokenizer,
            attack=attack,
            prompts=prompts,
            output_dir=output_dir,
            **kwargs,
        )
        all_results[method_name] = results

        # Clear memory between attacks
        gc.collect()
        torch.cuda.empty_cache()

    # Save combined summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_prompts": len(prompts),
        "methods": {},
    }

    for method_name, results in all_results.items():
        method_summary = {
            "attack_metadata": results.get("attack_metadata", {}),
        }
        # Add success rates from each methodology
        for key in results:
            if "success_rate" in key:
                method_summary[key] = results[key]
        summary["methods"][method_name] = method_summary

    summary_path = os.path.join(output_dir, "baseline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run jailbreak baseline attacks on OSS model"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["artprompt", "deepinception", "ffa", "all"],
        help="Attack method to run (default: all)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        help="Model path or HF model ID",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="harmful_test",
        help="Dataset to use: 'harmful_test', 'harmful_val', or a processed dataset name",
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=None,
        help="Number of prompts to use (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pipeline/runs/jailbreak_baselines",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--eval_methodologies",
        type=str,
        nargs="+",
        default=["substring_matching", "openai"],
        help="Evaluation methodologies to use",
    )
    parser.add_argument(
        "--use_unsloth",
        action="store_true",
        default=True,
        help="Use unsloth for model loading (default: True)",
    )
    parser.add_argument(
        "--no_unsloth",
        action="store_true",
        help="Don't use unsloth (use transformers)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="System prompt to use (default: None)",
    )

    args = parser.parse_args()

    # Handle unsloth flag
    use_unsloth = args.use_unsloth and not args.no_unsloth

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset in ["harmful_test", "harmful_val", "harmful_train"]:
        split = args.dataset.replace("harmful_", "")
        prompts = load_dataset_split("harmful", split)
    else:
        prompts = load_dataset(args.dataset)

    # Limit number of prompts if specified
    if args.n_prompts:
        prompts = prompts[:args.n_prompts]

    print(f"Loaded {len(prompts)} prompts")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save run config
    config = vars(args).copy()
    config["timestamp"] = timestamp
    config["n_prompts_actual"] = len(prompts)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load model
    model, tokenizer = load_model(args.model_path, use_unsloth=use_unsloth)

    # Run attacks
    if args.method == "all":
        results = run_all_baselines(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            output_dir=output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            system_prompt=args.system_prompt,
            eval_methodologies=args.eval_methodologies,
        )
    else:
        attack = get_attack_method(args.method)
        results = run_baseline_attack(
            model=model,
            tokenizer=tokenizer,
            attack=attack,
            prompts=prompts,
            output_dir=output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            system_prompt=args.system_prompt,
            eval_methodologies=args.eval_methodologies,
        )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
