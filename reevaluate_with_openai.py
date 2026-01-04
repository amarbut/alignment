#!/usr/bin/env python3
"""
Re-evaluate existing completions using OpenAI's GPT-4 for more accurate classification.

This script reads existing completion files and re-evaluates them with the OpenAI
judge to classify responses into: refusal, full_response, or non_response.
"""

import os
import json
import argparse
from pathlib import Path
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak


def reevaluate_completions(completions_dir: str, output_dir: str = None, delay: float = 1.0):
    """
    Re-evaluate all completion files in a directory with OpenAI classifier.

    Args:
        completions_dir: Directory containing *_completions.json files
        output_dir: Directory to save re-evaluated results (defaults to same as completions_dir)
        delay: Delay in seconds between API requests for rate limiting (default: 1.0)
    """
    if output_dir is None:
        output_dir = completions_dir

    completions_dir = Path(completions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all completion files
    completion_files = list(completions_dir.glob("*_completions.json"))

    if not completion_files:
        print(f"No completion files found in {completions_dir}")
        return

    print(f"Found {len(completion_files)} completion files to re-evaluate")

    for completion_file in completion_files:
        print(f"\n{'='*80}")
        print(f"Processing: {completion_file.name}")
        print(f"{'='*80}")

        # Load completions
        with open(completion_file, 'r') as f:
            completions = json.load(f)

        print(f"Loaded {len(completions)} completions")

        # Prepare output path
        dataset_name = completion_file.stem.replace('_completions', '')
        eval_output_path = output_dir / f"{dataset_name}_evaluations_openai.json"

        # Run OpenAI evaluation
        evaluation = evaluate_jailbreak(
            completions=completions,
            methodologies=["openai"],
            evaluation_path=str(eval_output_path),
            openai_delay=delay
        )

        print(f"\n{'='*80}")
        print(f"Results for {dataset_name}:")
        print(f"{'='*80}")
        print(f"OpenAI ASR (full_response rate): {evaluation['openai_success_rate']:.2%}")
        print(f"\nBreakdown:")
        for category, count in evaluation['openai_overall_counts'].items():
            percentage = count / len(completions) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")

        print(f"\nPer-category ASR:")
        for category, asr in evaluation['openai_per_category'].items():
            print(f"  {category}: {asr:.2%}")

        # Also save the completions with OpenAI annotations
        annotated_completions_path = output_dir / f"{dataset_name}_completions_openai.json"
        with open(annotated_completions_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation['completions'], f, indent=4, ensure_ascii=False)

        print(f"\nSaved annotated completions to: {annotated_completions_path}")
        print(f"Saved evaluations to: {eval_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate completions with OpenAI classifier"
    )
    parser.add_argument(
        "completions_dir",
        type=str,
        help="Directory containing *_completions.json files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save re-evaluated results (defaults to completions_dir)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between API requests for rate limiting (default: 1.0)"
    )

    args = parser.parse_args()

    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        return

    reevaluate_completions(args.completions_dir, args.output_dir, args.delay)


if __name__ == "__main__":
    main()
