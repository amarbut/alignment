#!/usr/bin/env python3
"""
GCG Transfer Attack: Optimize on few prompts, evaluate on many.

This script:
1. Optimizes adversarial suffixes on a small set of prompts
2. Selects the best suffix (lowest loss)
3. Creates a modified dataset with suffix appended to all prompts
4. Uses the standard pipeline for generation and evaluation

Usage:
    nohup python run_gcg_transfer.py \
        --optimize_dataset dataset/processed/gcg_optimize.json \
        --eval_dataset harmful_test \
        --n_eval 100 \
        > gcg_transfer.log 2>&1 &
"""

import argparse
import gc
import json
import os
from datetime import datetime
from typing import List, Dict, Any

import torch
from tqdm import tqdm

from dataset.load_dataset import load_dataset_split, load_dataset
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.model_utils.model_factory import construct_model_base
from jailbreak_baselines import GCGAttack, GCGConfig


def load_optimize_dataset(path: str) -> List[Dict[str, str]]:
    """Load the optimization dataset."""
    with open(path) as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, list):
        if isinstance(data[0], str):
            return [{"instruction": d, "category": "unknown"} for d in data]
        return data
    elif isinstance(data, dict) and "prompts" in data:
        return data["prompts"]

    raise ValueError(f"Could not parse optimization dataset from {path}")


def generate_and_evaluate_with_suffix(
    model_base,
    dataset: List[Dict],
    suffix: str,
    output_dir: str,
    label: str,
    eval_methodologies: List[str],
    max_new_tokens: int = 100,
):
    """
    Generate completions with adversarial suffix and evaluate.

    Similar to generate_and_evaluate_completions but appends suffix to prompts
    instead of using intervention hooks.
    """
    completions_dir = os.path.join(output_dir, 'completions', label)
    os.makedirs(completions_dir, exist_ok=True)

    # Create modified dataset with suffix appended
    modified_dataset = []
    for item in dataset:
        modified_dataset.append({
            'instruction': f"{item['instruction']} {suffix}",
            'category': item.get('category', 'unknown'),
            'original_instruction': item['instruction'],
        })

    # Generate completions using model_base (no hooks needed for GCG)
    print(f"\nGenerating completions with suffix...")
    completions = model_base.generate_completions(
        modified_dataset,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        max_new_tokens=max_new_tokens,
        batch_size=4,
    )

    # Restore original prompts for evaluation (judge should see original, not with suffix)
    for i, completion in enumerate(completions):
        completion['original_prompt'] = dataset[i]['instruction']
        completion['adversarial_prompt'] = completion['prompt']
        completion['prompt'] = dataset[i]['instruction']  # Use original for judging
        completion['suffix'] = suffix

    # Save completions
    completions_path = os.path.join(completions_dir, f"completions.json")
    with open(completions_path, "w", encoding="utf-8") as f:
        json.dump(completions, f, indent=4, ensure_ascii=False)
    print(f"Saved completions to: {completions_path}")

    # Evaluate
    print(f"Evaluating completions...")
    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(completions_dir, "evaluations.json"),
    )

    # Save evaluations
    eval_path = os.path.join(completions_dir, "evaluations.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=4, ensure_ascii=False)
    print(f"Saved evaluations to: {eval_path}")

    return completions, evaluation


def main():
    parser = argparse.ArgumentParser(
        description="GCG Transfer Attack: Optimize on few, evaluate on many"
    )

    # Dataset settings
    parser.add_argument(
        "--optimize_dataset",
        type=str,
        default="dataset/processed/gcg_optimize.json",
        help="Path to optimization dataset (small, ~5 prompts)",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="harmful_test",
        help="Evaluation dataset name or path",
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=None,
        help="Number of evaluation prompts (default: all)",
    )

    # Model settings
    parser.add_argument(
        "--model_alias",
        type=str,
        default="oss_unsloth",
        choices=["oss", "oss_unsloth"],
        help="Model alias for model_factory",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        help="Model path (for direct loading during optimization)",
    )

    # GCG optimization settings
    parser.add_argument("--num_steps", type=int, default=250,
                        help="GCG optimization steps per prompt")
    parser.add_argument("--search_width", type=int, default=256)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--suffix_length", type=int, default=20,
                        help="Number of tokens in adversarial suffix")
    parser.add_argument("--seed", type=int, default=42)

    # Suffix selection
    parser.add_argument(
        "--suffix_strategy",
        type=str,
        default="best",
        choices=["best", "all"],
        help="'best' uses lowest-loss suffix, 'all' evaluates each suffix",
    )

    # Resume from previous optimization
    parser.add_argument(
        "--optimization_results",
        type=str,
        default=None,
        help="Path to existing optimization_results.json to skip optimization phase",
    )

    # Generation/evaluation settings
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument(
        "--eval_methodologies",
        type=str,
        nargs="+",
        default=["substring_matching", "openai"],
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pipeline/runs/gcg_transfer",
    )

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config = vars(args).copy()
    config["timestamp"] = timestamp
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # =========================================================================
    # Phase 1: Optimize suffixes (or load from previous run)
    # =========================================================================

    if args.optimization_results:
        # Load existing optimization results
        print(f"\n{'='*60}")
        print(f"Phase 1: Loading existing optimization results")
        print("="*60)

        with open(args.optimization_results) as f:
            optimization_results = json.load(f)
        print(f"Loaded {len(optimization_results)} optimized suffixes from {args.optimization_results}")
        for r in optimization_results:
            print(f"  loss={r['loss']:.4f}  prompt={r['prompt'][:50]}...")

    else:
        # Run optimization
        print(f"\n{'='*60}")
        print("Phase 1: Loading datasets and optimizing suffixes")
        print("="*60)

        optimize_prompts = load_optimize_dataset(args.optimize_dataset)
        print(f"Optimization prompts: {len(optimize_prompts)}")
        for i, p in enumerate(optimize_prompts):
            print(f"  {i+1}. {p['instruction'][:60]}...")

        # Load model for GCG optimization (using unsloth directly)
        print("\nLoading model for optimization...")
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        model.eval()
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create GCG config
        optim_str_init = " ".join(["!"] * args.suffix_length)
        gcg_config = GCGConfig(
            num_steps=args.num_steps,
            search_width=args.search_width,
            topk=args.topk,
            batch_size=args.batch_size,
            seed=args.seed,
            optim_str_init=optim_str_init,
        )

        attack = GCGAttack(
            model=model,
            tokenizer=tokenizer,
            config=gcg_config,
            use_nanogcg=False,
        )

        # Optimize suffixes
        optimization_results = []
        for i, prompt_dict in enumerate(optimize_prompts):
            prompt = prompt_dict["instruction"]
            print(f"\n[{i+1}/{len(optimize_prompts)}] Optimizing: {prompt[:60]}...")

            suffix, loss, history = attack.optimize_suffix(prompt)

            optimization_results.append({
                "prompt": prompt,
                "suffix": suffix,
                "loss": loss,
                "category": prompt_dict.get("category", "unknown"),
            })
            print(f"  Loss: {loss:.4f}")
            print(f"  Suffix: {suffix[:60]}...")

            gc.collect()
            torch.cuda.empty_cache()

        # Save optimization results
        with open(os.path.join(output_dir, "optimization_results.json"), "w") as f:
            json.dump(optimization_results, f, indent=2)

        # Clean up GCG model to free memory
        del model, tokenizer, attack
        gc.collect()
        torch.cuda.empty_cache()

    # Select suffix based on strategy
    if args.suffix_strategy == "best":
        best_result = min(optimization_results, key=lambda x: x["loss"])
        selected_suffixes = [{"suffix": best_result["suffix"],
                             "loss": best_result["loss"],
                             "source_prompt": best_result["prompt"]}]
        print(f"\nSelected best suffix (loss={best_result['loss']:.4f})")
    else:  # "all"
        selected_suffixes = [{"suffix": r["suffix"],
                             "loss": r["loss"],
                             "source_prompt": r["prompt"]}
                           for r in optimization_results]
        print(f"\nUsing all {len(selected_suffixes)} suffixes")

    # =========================================================================
    # Phase 2: Load evaluation dataset and model_base
    # =========================================================================
    print(f"\n{'='*60}")
    print("Phase 2: Loading evaluation model and dataset")
    print("="*60)

    # Load eval dataset
    if args.eval_dataset in ["harmful_test", "harmful_val", "harmful_train"]:
        split = args.eval_dataset.replace("harmful_", "")
        eval_dataset = load_dataset_split("harmful", split)
    else:
        eval_dataset = load_dataset(args.eval_dataset)

    if args.n_eval:
        eval_dataset = eval_dataset[:args.n_eval]
    print(f"Evaluation prompts: {len(eval_dataset)}")

    # Load model using model_factory for consistent generation
    print("\nLoading model for evaluation...")
    model_base = construct_model_base(args.model_path)

    # =========================================================================
    # Phase 3: Generate and evaluate with each suffix
    # =========================================================================
    print(f"\n{'='*60}")
    print("Phase 3: Generating and evaluating completions")
    print("="*60)

    all_results = {}

    for i, suffix_info in enumerate(selected_suffixes):
        suffix = suffix_info["suffix"]
        label = f"suffix_{i}" if len(selected_suffixes) > 1 else "gcg_transfer"

        print(f"\n--- {label} (loss={suffix_info['loss']:.4f}) ---")

        completions, evaluation = generate_and_evaluate_with_suffix(
            model_base=model_base,
            dataset=eval_dataset,
            suffix=suffix,
            output_dir=output_dir,
            label=label,
            eval_methodologies=args.eval_methodologies,
            max_new_tokens=args.max_new_tokens,
        )

        all_results[label] = {
            "suffix": suffix,
            "source_prompt": suffix_info["source_prompt"],
            "source_loss": suffix_info["loss"],
            "evaluation": {k: v for k, v in evaluation.items() if k != "completions"},
        }

        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # Phase 4: Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("Results Summary")
    print("="*60)

    for label, result in all_results.items():
        print(f"\n{label}:")
        print(f"  Source prompt: {result['source_prompt'][:50]}...")
        print(f"  Source loss: {result['source_loss']:.4f}")
        for key, value in result["evaluation"].items():
            if "success_rate" in key or "asr" in key.lower():
                print(f"  {key}: {value:.4f}")

    # Save final summary
    summary = {
        "config": config,
        "optimization_results": optimization_results,
        "num_eval_prompts": len(eval_dataset),
        "results": all_results,
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Complete! Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
