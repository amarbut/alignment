"""
Expert Activation Frequency Diff Generator

Generates expert activation frequency diffs (harmful vs harmless) using
top-1 routing. Output format is 4-column:
    [expert_id, diff_pct, harmful_pct, harmless_pct]
where diff_pct = harmful_pct - harmless_pct.

Use this to regenerate any expert_diffs files that were saved in the older
2-column format [expert_id, diff_pct] and are missing raw frequencies.

Usage:
    python run_expert_diffs.py \\
        --model_path allenai/OLMoE-1B-7B-0924-Instruct \\
        --system_prompt none

    python run_expert_diffs.py \\
        --model_path allenai/OLMoE-1B-7B-0924-Instruct \\
        --system_prompt llama_2
"""

# =============================================================================
# Add parent directory to path so imports resolve from refusal_steering/
# =============================================================================
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# Set HF cache BEFORE any HuggingFace imports
# =============================================================================
import argparse as _argparse_early

_early_parser = _argparse_early.ArgumentParser(add_help=False)
_early_parser.add_argument('--model_path', type=str, default='')
_early_args, _ = _early_parser.parse_known_args()

if _early_args.model_path:
    from model_utils.hf_cache_config import set_hf_cache_from_path
    set_hf_cache_from_path(_early_args.model_path)
# =============================================================================

import argparse

from model_utils.model_factory_moe import construct_model_base
from model_utils.model_card_factory import create_model_card
from submodules.expert_diff_generator import generate_expert_diffs_for_model


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate expert activation frequency diffs (harmful vs harmless)"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the model'
    )
    parser.add_argument(
        '--system_prompt',
        type=str,
        default='lightweight',
        choices=['none', 'llama_2', 'lightweight'],
        help='System prompt to use (default: lightweight)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Output JSON path (default: expert_diffs/sys_prompt_{sp}/{model}_expert_diffs.json)'
    )
    parser.add_argument(
        '--harmful_dataset',
        type=str,
        default='dataset/splits/harmful_train.json',
        help='Path to harmful prompts JSON'
    )
    parser.add_argument(
        '--harmless_dataset',
        type=str,
        default='dataset/splits/harmless_train.json',
        help='Path to harmless prompts JSON'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for processing (default: 4)'
    )
    parser.add_argument(
        '--last_n_tokens',
        type=int,
        default=5,
        help='Number of tokens from end of prompt to analyze (default: 5)'
    )
    parser.add_argument(
        '--num_harmful',
        type=int,
        default=None,
        help='Number of harmful samples to use (default: all)'
    )
    parser.add_argument(
        '--num_harmless',
        type=int,
        default=200,
        help='Number of harmless samples to use (default: 200)'
    )
    return parser.parse_args()


def run(args):
    print("=" * 80)
    print("EXPERT DIFF GENERATOR")
    print("=" * 80)
    print(f"Model:         {args.model_path}")
    print(f"System prompt: {args.system_prompt}")
    print("=" * 80)

    print("\nLoading model...")
    model_base = construct_model_base(args.model_path, system_prompt=args.system_prompt)
    model_base.model.config.pad_token_id = model_base.tokenizer.pad_token_id

    model_card = create_model_card(model_base)
    print(f"Model card:    {type(model_card).__name__}")
    print(f"Routing mode:  {model_card.get_expert_routing_mode()}")

    if args.output_path is None:
        output_path = os.path.join(
            'expert_diffs',
            f'sys_prompt_{args.system_prompt}',
            model_card.get_expert_diffs_filename(),
        )
    else:
        output_path = args.output_path

    print(f"Output path:   {output_path}")

    generate_expert_diffs_for_model(
        model_base=model_base,
        model_card=model_card,
        harmful_dataset_path=args.harmful_dataset,
        harmless_dataset_path=args.harmless_dataset,
        output_path=output_path,
        batch_size=args.batch_size,
        last_n_tokens=args.last_n_tokens,
        num_harmful=args.num_harmful,
        num_harmless=args.num_harmless,
        use_top_k=False,
    )

    print("\nDone.")


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
