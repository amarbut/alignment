"""
Evaluate fine-tuned MoE checkpoints for refusal behavior.

This script evaluates checkpoints from expert dropout fine-tuning experiments,
assessing both baseline refusal behavior and robustness to expert-specific attacks.

Usage:
    # Evaluate specific checkpoints
    python evaluate_finetuned_checkpoints.py \
        --checkpoint_dirs checkpoints/3kX2e_expert_dropout_0.0/checkpoint-1500 \
                         checkpoints/3kX2e_expert_dropout_0.2/checkpoint-1000

    # Evaluate all recommended checkpoints
    python evaluate_finetuned_checkpoints.py --eval_recommended

    # Include intervention testing
    python evaluate_finetuned_checkpoints.py --eval_recommended --test_intervention
"""

import os
import sys
import torch
import json
import random
import argparse
from pathlib import Path
from typing import List, Optional, Dict

# Add alignment directory to path
alignment_dir = os.path.dirname(os.path.abspath(__file__))
if alignment_dir not in sys.path:
    sys.path.insert(0, alignment_dir)

import fix_hf_cache
from unsloth import FastLanguageModel
from peft import PeftModel

from dataset.load_dataset import load_dataset_split, load_dataset
from pipeline.config import Config
from pipeline.model_utils.oss_model import OSSModel, tokenize_instructions_oss_chat
from pipeline.submodules.select_direction import get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak

# Try to import expert-specific functions if available
try:
    from expert_intervention import get_expert_weighted_intervention_hooks
    EXPERT_INTERVENTION_AVAILABLE = True
except ImportError:
    EXPERT_INTERVENTION_AVAILABLE = False
    print("Warning: expert_intervention module not available, skipping intervention tests")


# Final checkpoints for direct dropout comparison
FINAL_CHECKPOINTS = {
    "dropout_0.0": "checkpoints/3kX2e_expert_dropout_0.0/final",
    "dropout_0.2": "checkpoints/3kX2e_expert_dropout_0.2/final",
    "dropout_0.3": "checkpoints/3kX2e_expert_dropout_0.3/final",
    "dropout_0.4": "checkpoints/3kX2e_expert_dropout_0.4/final",
    "dropout_0.5": "checkpoints/3kX2e_expert_dropout_0.5/final",
}

# Best intermediate checkpoints based on training loss analysis
# (for comparison - captures model before divergence)
BEST_CHECKPOINTS = {
    "dropout_0.0": "checkpoints/3kX2e_expert_dropout_0.0/checkpoint-1500",
    "dropout_0.2": "checkpoints/3kX2e_expert_dropout_0.2/checkpoint-1000",
    "dropout_0.3": "checkpoints/3kX2e_expert_dropout_0.3/checkpoint-1000",
    "dropout_0.4": "checkpoints/3kX2e_expert_dropout_0.4/checkpoint-500",
    "dropout_0.5": "checkpoints/3kX2e_expert_dropout_0.5/checkpoint-500",
}


class UnslothOSSModelWrapper:
    """
    Wrapper for unsloth models to make them compatible with the evaluation pipeline.
    Mimics the OSSModel interface while using unsloth/PEFT models.
    """

    def __init__(self, base_model_name: str, adapter_path: Optional[str] = None, max_seq_length: int = 512):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.max_seq_length = max_seq_length

        # Load model
        if adapter_path:
            print(f"\nLoading base model: {base_model_name}")
            print(f"Loading adapter from: {adapter_path}")

            base_model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_name,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )

            # Load PEFT adapter
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
            self.tokenizer = tokenizer
            print("✓ Loaded model with LoRA adapter")
        else:
            print(f"\nLoading base model: {base_model_name}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_name,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            print("✓ Loaded base model")

        # Set up tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set refusal tokens (from OSSModel)
        self.refusal_toks = [40, 357, 2305, 1877, 33680, 47483]  # 'I', 'As', 'Sorry'
        self.refusal_score_suffix_toks = None

        # Set tokenization function
        self.tokenize_instructions_fn = self._tokenize_instructions

    def _tokenize_instructions(self, instructions, **kwargs):
        """Tokenize instructions using OSS chat template."""
        return tokenize_instructions_oss_chat(
            self.tokenizer,
            instructions,
            **kwargs
        )

    def generate_completions(
        self,
        instructions: List[str],
        fwd_pre_hooks=None,
        fwd_hooks=None,
        max_new_tokens: int = 256,
        batch_size: int = 8
    ) -> List[Dict[str, str]]:
        """Generate completions for instructions."""

        completions = []

        # Register hooks if provided
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
            # Process in batches
            for i in range(0, len(instructions), batch_size):
                batch_instructions = instructions[i:i+batch_size]

                # Tokenize
                tokenized = self.tokenize_instructions_fn(
                    batch_instructions,
                    include_trailing_whitespace=True
                )
                input_ids = tokenized['input_ids'].to(self.model.device)
                attention_mask = tokenized['attention_mask'].to(self.model.device)

                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                # Decode
                for j, output in enumerate(outputs):
                    # Remove input tokens
                    prompt_length = input_ids[j].shape[0]
                    completion_ids = output[prompt_length:]
                    completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

                    completions.append({
                        "prompt": batch_instructions[j],
                        "response": completion,
                        "category": "unknown"  # Will be set by caller if needed
                    })

        finally:
            # Remove hooks
            for handle in hook_handles:
                handle.remove()

        return completions


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned MoE checkpoints"
    )

    parser.add_argument(
        '--base_model',
        type=str,
        default='unsloth/gpt-oss-20b-unsloth-bnb-4bit',
        help='Base model name'
    )

    parser.add_argument(
        '--checkpoint_dirs',
        type=str,
        nargs='+',
        help='List of checkpoint directories to evaluate'
    )

    parser.add_argument(
        '--eval_final',
        action='store_true',
        help='Evaluate final checkpoints for direct dropout comparison'
    )

    parser.add_argument(
        '--eval_best',
        action='store_true',
        help='Evaluate best intermediate checkpoints (before divergence)'
    )

    parser.add_argument(
        '--test_intervention',
        action='store_true',
        help='Test robustness with expert-specific interventions'
    )

    parser.add_argument(
        '--intervention_layer',
        type=int,
        default=None,
        help='Layer for intervention (required if --test_intervention)'
    )

    parser.add_argument(
        '--intervention_expert',
        type=int,
        default=None,
        help='Expert ID for intervention (required if --test_intervention)'
    )

    parser.add_argument(
        '--direction_path',
        type=str,
        default=None,
        help='Path to intervention direction tensor (required if --test_intervention)'
    )

    parser.add_argument(
        '--n_test',
        type=int,
        default=100,
        help='Number of test samples'
    )

    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=256,
        help='Maximum new tokens for generation'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for results'
    )

    parser.add_argument(
        '--eval_baseline_only',
        action='store_true',
        help='Only evaluate baseline (no dataset) - for unsloth base model comparison'
    )

    return parser.parse_args()


def load_and_sample_datasets(n_test: int):
    """Load and sample test datasets."""
    random.seed(42)

    # Load test sets
    harmful_test = random.sample(
        load_dataset_split(harmtype='harmful', split='test', instructions_only=True),
        min(n_test, len(load_dataset_split(harmtype='harmful', split='test', instructions_only=True)))
    )
    harmless_test = random.sample(
        load_dataset_split(harmtype='harmless', split='test', instructions_only=True),
        min(n_test, len(load_dataset_split(harmtype='harmless', split='test', instructions_only=True)))
    )

    return harmful_test, harmless_test


def evaluate_checkpoint_baseline(
    model_wrapper: UnslothOSSModelWrapper,
    harmful_test: List[str],
    harmless_test: List[str],
    output_dir: str,
    checkpoint_label: str,
    max_new_tokens: int = 256
):
    """
    Evaluate checkpoint baseline performance (no intervention).

    Returns:
        Dict with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING BASELINE: {checkpoint_label}")
    print(f"{'='*80}")

    # Create output directory
    completions_dir = os.path.join(output_dir, checkpoint_label, 'baseline')
    os.makedirs(completions_dir, exist_ok=True)

    results = {}

    # Evaluate on harmful prompts
    print("\n--- Harmful Prompts ---")
    harmful_completions = model_wrapper.generate_completions(
        harmful_test,
        max_new_tokens=max_new_tokens
    )

    # Save completions
    harmful_path = os.path.join(completions_dir, 'harmful_completions.json')
    with open(harmful_path, 'w', encoding='utf-8') as f:
        json.dump(harmful_completions, f, indent=4, ensure_ascii=False)
    print(f"Saved to: {harmful_path}")

    # Evaluate
    harmful_eval = evaluate_jailbreak(
        completions=harmful_completions,
        methodologies=("substring_matching", "llamaguard2", "openai"),
        evaluation_path=os.path.join(completions_dir, 'harmful_evaluations.json')
    )
    results['harmful'] = harmful_eval

    # Evaluate on harmless prompts
    print("\n--- Harmless Prompts ---")
    harmless_completions = model_wrapper.generate_completions(
        harmless_test,
        max_new_tokens=max_new_tokens
    )

    # Save completions
    harmless_path = os.path.join(completions_dir, 'harmless_completions.json')
    with open(harmless_path, 'w', encoding='utf-8') as f:
        json.dump(harmless_completions, f, indent=4, ensure_ascii=False)
    print(f"Saved to: {harmless_path}")

    # Evaluate
    harmless_eval = evaluate_jailbreak(
        completions=harmless_completions,
        methodologies=("substring_matching", "openai"),
        evaluation_path=os.path.join(completions_dir, 'harmless_evaluations.json')
    )
    results['harmless'] = harmless_eval

    # Compute summary metrics
    # Note: evaluate_jailbreak returns "success_rate" = jailbreak success (ASR)
    # For harmful prompts: refusal_rate = 1 - ASR
    # For harmless prompts: compliance_rate = ASR (model complied with harmless request)
    harmful_asr = harmful_eval.get('substring_matching_success_rate', 0)
    harmless_asr = harmless_eval.get('substring_matching_success_rate', 0)

    summary = {
        'checkpoint': checkpoint_label,
        'harmful_refusal_rate': 1 - harmful_asr,  # Refusal = 1 - jailbreak success
        'harmless_compliance_rate': harmless_asr,  # For harmless, ASR = compliance
        'harmful_asr': harmful_asr,  # Also track raw ASR
        'n_harmful': len(harmful_test),
        'n_harmless': len(harmless_test)
    }

    # Save summary
    summary_path = os.path.join(completions_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n✓ Summary:")
    print(f"  Harmful refusal rate: {summary['harmful_refusal_rate']:.2%}")
    print(f"  Harmless compliance rate: {summary['harmless_compliance_rate']:.2%}")

    return results, summary


def main():
    args = parse_arguments()

    # Determine checkpoints to evaluate
    if args.eval_final:
        checkpoint_dirs = list(FINAL_CHECKPOINTS.values())
        checkpoint_labels = list(FINAL_CHECKPOINTS.keys())
    elif args.eval_best:
        checkpoint_dirs = list(BEST_CHECKPOINTS.values())
        checkpoint_labels = list(BEST_CHECKPOINTS.keys())
    elif args.checkpoint_dirs:
        checkpoint_dirs = args.checkpoint_dirs
        checkpoint_labels = [Path(d).parent.name + "_" + Path(d).name for d in checkpoint_dirs]
    else:
        print("Error: Must specify --checkpoint_dirs, --eval_final, or --eval_best")
        return

    # Also evaluate baseline model
    if args.eval_baseline_only:
        checkpoint_dirs = [None] + checkpoint_dirs
        checkpoint_labels = ['baseline_unsloth'] + checkpoint_labels

    # Load datasets
    print("\n" + "="*80)
    print("LOADING TEST DATASETS")
    print("="*80)
    harmful_test, harmless_test = load_and_sample_datasets(args.n_test)
    print(f"Loaded {len(harmful_test)} harmful and {len(harmless_test)} harmless prompts")

    # Evaluate each checkpoint
    all_summaries = []

    for checkpoint_dir, checkpoint_label in zip(checkpoint_dirs, checkpoint_labels):
        print(f"\n\n{'='*80}")
        print(f"CHECKPOINT: {checkpoint_label}")
        print(f"{'='*80}")

        # Load model
        model_wrapper = UnslothOSSModelWrapper(
            base_model_name=args.base_model,
            adapter_path=checkpoint_dir
        )

        # Evaluate baseline
        results, summary = evaluate_checkpoint_baseline(
            model_wrapper,
            harmful_test,
            harmless_test,
            args.output_dir,
            checkpoint_label,
            max_new_tokens=args.max_new_tokens
        )

        all_summaries.append(summary)

        # Clean up to free memory
        del model_wrapper
        torch.cuda.empty_cache()

    # Save comparison
    comparison_path = os.path.join(args.output_dir, 'checkpoint_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(all_summaries, f, indent=4)

    print(f"\n\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_dir}")
    print(f"\nCheckpoint Comparison:")
    print(f"{'Checkpoint':<40} {'Harmful Refusal':<20} {'Harmless Compliance':<20}")
    print("-" * 80)
    for s in all_summaries:
        print(f"{s['checkpoint']:<40} {s['harmful_refusal_rate']:>18.2%} {s['harmless_compliance_rate']:>18.2%}")


if __name__ == "__main__":
    main()
