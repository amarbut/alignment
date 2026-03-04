"""
Capability benchmarks for baseline and expert-steered models.

Evaluates MMLU, HellaSwag, ARC, Winogrande, GSM8K, and TruthfulQA using
lm-evaluation-harness. Loads the model once, runs baseline evaluation, then
registers the intervention hooks (from the threshold pipeline's best expert),
re-runs the same tasks, and saves a side-by-side comparison.

The intervention hooks are loaded from the threshold run's experiment_metadata.json
and the shared direction cache — no recomputation needed.

Note on TinyHellaSwag: the harness has no separate tiny task; use --hellaswag_limit
to cap the number of examples (default: 1000, matching common "tiny" usage).

Requirements:
    pip install lm-eval>=0.4.0
    # or: pip install lm-eval[math,sentencepiece]  (for GSM8K / some MMLU subtasks)

Usage:
    # Both baseline and intervened (default):
    python run_benchmarks.py \\
        --model_path allenai/OLMoE-1B-7B-0924-Instruct \\
        --system_prompt lightweight

    # Baseline only:
    python run_benchmarks.py --model_path ... --system_prompt lightweight --mode baseline

    # Point at a specific metadata file instead of auto-locating:
    python run_benchmarks.py \\
        --model_path ... --system_prompt lightweight \\
        --metadata_path runs/.../experiment_metadata.json
"""

# =============================================================================
# Add parent directory to path so imports resolve from refusal_steering/
# =============================================================================
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# Check lm_eval is installed before doing anything expensive
# =============================================================================
try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
except ImportError:
    print(
        "ERROR: lm-evaluation-harness is not installed.\n"
        "Install it with:\n"
        "    pip install lm-eval>=0.4.0\n"
        "or for full benchmark support (GSM8K, some MMLU subtasks):\n"
        "    pip install lm-eval[math,sentencepiece]"
    )
    sys.exit(1)

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
import json
import torch

from config import Config
from model_utils.model_factory_moe import construct_model_base
from submodules.expert_steering.expert_intervention import get_expert_weighted_intervention_hooks
from run_expert_steering import _get_shared_cache_dir, _load_or_compute_layer_directions


# =============================================================================
# Task configuration
# =============================================================================

# Standard few-shot counts matching Open LLM Leaderboard / common practice.
# Set to 0 for all if you only care about the relative baseline vs intervened gap.
TASK_CONFIG = {
    "mmlu":             {"num_fewshot": 5},
    "hellaswag":        {"num_fewshot": 10},
    "arc_easy":         {"num_fewshot": 25},
    "arc_challenge":    {"num_fewshot": 25},
    "winogrande":       {"num_fewshot": 5},
    "gsm8k":            {"num_fewshot": 5},
    "truthfulqa_mc1":   {"num_fewshot": 0},
}

# Primary metric to extract per task (lm_eval metric key)
TASK_METRIC = {
    "mmlu":             "acc,none",
    "hellaswag":        "acc_norm,none",
    "arc_easy":         "acc_norm,none",
    "arc_challenge":    "acc_norm,none",
    "winogrande":       "acc,none",
    "gsm8k":            "exact_match,strict-match",
    "truthfulqa_mc1":   "acc,none",
}


def extract_metric(results, task, metric_key):
    """Pull the primary metric from lm_eval results dict."""
    task_results = results.get("results", {}).get(task, {})
    return task_results.get(metric_key)


# =============================================================================
# Hook helpers
# =============================================================================

def register_hooks(fwd_pre_hooks, fwd_hooks):
    """Register hooks permanently on their modules; return handles for removal."""
    handles = []
    for module, hook_fn in fwd_pre_hooks:
        handles.append(module.register_forward_pre_hook(hook_fn))
    for module, hook_fn in fwd_hooks:
        handles.append(module.register_forward_hook(hook_fn))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# =============================================================================
# Argument parsing
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Capability benchmarks: baseline vs expert-steered model"
    )
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--system_prompt', type=str, default=None,
                        choices=['none', 'llama_2', 'lightweight'])
    parser.add_argument('--mode', type=str, default='both',
                        choices=['baseline', 'intervened', 'both'],
                        help='Which configurations to evaluate (default: both)')

    # Locating the intervention settings
    parser.add_argument('--metadata_path', type=str, default=None,
                        help='Path to experiment_metadata.json from a threshold run. '
                             'Auto-located under --runs_dir if not provided.')
    parser.add_argument('--runs_dir', type=str, default='runs',
                        help='Root runs directory (default: runs/)')

    # Task selection
    parser.add_argument('--tasks', type=str, nargs='+',
                        default=list(TASK_CONFIG.keys()),
                        help='Tasks to evaluate (default: all)')
    parser.add_argument('--hellaswag_limit', type=int, default=1000,
                        help='Max hellaswag examples — simulates TinyHellaSwag (default: 1000)')
    parser.add_argument('--zero_shot', action='store_true',
                        help='Override all tasks to 0-shot')

    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for lm_eval (default: 1)')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Max context length for lm_eval (default: 2048; reduce if OOM)')

    # Direction loading
    parser.add_argument('--force_generate', action='store_true',
                        help='Recompute cached directions even if they exist')
    parser.add_argument('--batch_size_directions', type=int, default=32,
                        help='Batch size for direction computation if needed (default: 32)')

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save results JSON. Default: alongside metadata.')

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_arguments()

    print("=" * 80)
    print("CAPABILITY BENCHMARKS")
    print("=" * 80)
    print(f"Model:  {args.model_path}")
    print(f"Mode:   {args.mode}")
    print(f"Tasks:  {args.tasks}")
    print("=" * 80)

    base_model_name = os.path.basename(args.model_path)

    # ------------------------------------------------------------------
    # Locate experiment metadata (needed for intervened mode)
    # ------------------------------------------------------------------
    if args.metadata_path:
        metadata_path = args.metadata_path
    else:
        # Resolve system_prompt via Config default if not supplied
        sp = args.system_prompt or Config().system_prompt
        metadata_path = os.path.join(
            args.runs_dir, base_model_name,
            "expert_steering_threshold",
            f"sys_prompt_{sp}",
            "grid_search", "experiment_metadata.json",
        )

    metadata = None
    if args.mode in ('intervened', 'both'):
        if not os.path.exists(metadata_path):
            print(f"ERROR: experiment metadata not found at:\n  {metadata_path}")
            print("Run the threshold pipeline first, or pass --metadata_path.")
            sys.exit(1)
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"\nIntervention settings (from {metadata_path}):")
        print(f"  Layer:    {metadata['layer']}")
        print(f"  Expert:   {metadata['expert']}")
        print(f"  Position: {metadata['position']}")
        print(f"  Coeff:    {metadata['coeff']}")

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.dirname(metadata_path) if metadata_path else "."
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    # Load without system prompt so benchmarks use the model's default behaviour
    model_base = construct_model_base(args.model_path, system_prompt="none")

    from model_utils.model_card_factory import create_model_card
    model_card = create_model_card(model_base)

    # ------------------------------------------------------------------
    # Load direction and build hooks (intervened mode only)
    # ------------------------------------------------------------------
    fwd_pre_hooks, fwd_hooks = [], []

    if args.mode in ('intervened', 'both'):
        print("\n" + "=" * 80)
        print("LOADING INTERVENTION DIRECTION")
        print("=" * 80)

        # Use system_prompt from metadata path, not from model load
        sp = args.system_prompt or Config().system_prompt
        cache_dir = _get_shared_cache_dir(base_model_name, sp)
        layer_idx  = metadata['layer']
        expert_id  = metadata['expert']
        position   = metadata['position']
        coeff      = metadata['coeff']

        layer_cache = _load_or_compute_layer_directions(
            layer_idx=layer_idx,
            expert_ids_needed=[expert_id],
            model_base=model_base,
            harmful_train=[],   # won't compute — direction should already be cached
            harmless_train=[],
            batch_size=args.batch_size_directions,
            cache_dir=cache_dir,
            force_generate=args.force_generate,
        )

        if expert_id not in layer_cache:
            print(f"ERROR: direction for L{layer_idx} E{expert_id} not in cache at {cache_dir}.")
            print("Run the threshold pipeline first to populate the direction cache.")
            sys.exit(1)

        mean_diff = layer_cache[expert_id]
        direction = mean_diff[position].to(model_base.model.device, model_base.model.dtype)

        fwd_pre_hooks, fwd_hooks = get_expert_weighted_intervention_hooks(
            model_base,
            layer_idx=layer_idx,
            expert_id=expert_id,
            direction=direction,
            coeff=coeff,
            model_card=model_card,
        )
        print(f"  Direction loaded — L{layer_idx} E{expert_id} pos={position} coeff={coeff}")

    # ------------------------------------------------------------------
    # Build lm_eval wrapper (shared between baseline and intervened)
    # ------------------------------------------------------------------
    # Temporarily clear the chat template so lm_eval sees plain text prompts
    original_chat_template = model_base.tokenizer.chat_template
    model_base.tokenizer.chat_template = None

    lm = HFLM(
        pretrained=model_base.model,
        tokenizer=model_base.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Build task kwargs (num_fewshot, per-task limits)
    tasks_to_run = [t for t in args.tasks if t in TASK_CONFIG]
    unknown = [t for t in args.tasks if t not in TASK_CONFIG]
    if unknown:
        print(f"Warning: unknown tasks will be passed through as-is: {unknown}")
        tasks_to_run += unknown

    def make_eval_kwargs():
        kwargs = {}
        for task in tasks_to_run:
            nfs = 0 if args.zero_shot else TASK_CONFIG.get(task, {}).get("num_fewshot", 0)
            limit = args.hellaswag_limit if task == "hellaswag" else None
            kwargs[task] = {"num_fewshot": nfs}
            if limit is not None:
                kwargs[task]["limit"] = limit
        return kwargs

    task_kwargs = make_eval_kwargs()

    all_results = {}

    # ------------------------------------------------------------------
    # Baseline evaluation
    # ------------------------------------------------------------------
    if args.mode in ('baseline', 'both'):
        print("\n" + "=" * 80)
        print("RUNNING BASELINE BENCHMARKS")
        print("=" * 80)

        baseline_results = _run_tasks(lm, tasks_to_run, task_kwargs, label="baseline")
        all_results['baseline'] = baseline_results

    # ------------------------------------------------------------------
    # Intervened evaluation
    # ------------------------------------------------------------------
    if args.mode in ('intervened', 'both'):
        print("\n" + "=" * 80)
        print("RUNNING INTERVENED BENCHMARKS")
        print("=" * 80)

        handles = register_hooks(fwd_pre_hooks, fwd_hooks)
        try:
            intervened_results = _run_tasks(lm, tasks_to_run, task_kwargs, label="intervened")
            all_results['intervened'] = intervened_results
        finally:
            remove_hooks(handles)

    # Restore chat template
    model_base.tokenizer.chat_template = original_chat_template

    # ------------------------------------------------------------------
    # Print and save comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    _print_table(all_results, tasks_to_run)

    output_path = os.path.join(out_dir, "benchmark_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to: {output_path}")


# =============================================================================
# Task runner (per-task control over num_fewshot / limit)
# =============================================================================

def _run_tasks(lm, tasks, task_kwargs, label):
    """
    Run each task individually so we can pass per-task num_fewshot and limit.
    Returns a flat dict: {task_name: {metric_key: value, ...}}.
    """
    combined = {}
    for task in tasks:
        kwargs = task_kwargs.get(task, {})
        num_fewshot = kwargs.get("num_fewshot", 0)
        limit = kwargs.get("limit", None)

        print(f"\n  [{label}] {task}  (fewshot={num_fewshot}"
              + (f", limit={limit}" if limit else "") + ")")

        result = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=num_fewshot,
            limit=limit,
            verbosity="WARNING",
        )
        task_metrics = result.get("results", {}).get(task, {})
        combined[task] = task_metrics

        # Print primary metric immediately
        primary_key = TASK_METRIC.get(task)
        if primary_key and primary_key in task_metrics:
            print(f"    → {primary_key}: {task_metrics[primary_key]:.4f}")

    return combined


# =============================================================================
# Comparison table printer
# =============================================================================

def _print_table(all_results, tasks):
    modes = list(all_results.keys())
    col_w = 14

    # Header
    header = f"  {'Task':<22}" + "".join(f"{m:>{col_w}}" for m in modes)
    if len(modes) == 2:
        header += f"{'Δ (int-base)':>{col_w}}"
    print(header)
    print("  " + "-" * (22 + col_w * (len(modes) + (1 if len(modes) == 2 else 0))))

    for task in tasks:
        row = f"  {task:<22}"
        vals = []
        for mode in modes:
            metric_key = TASK_METRIC.get(task)
            v = all_results[mode].get(task, {}).get(metric_key) if metric_key else None
            vals.append(v)
            row += f"{f'{v:.4f}' if v is not None else 'N/A':>{col_w}}"

        if len(modes) == 2 and all(v is not None for v in vals):
            delta = vals[1] - vals[0]
            sign = "+" if delta >= 0 else ""
            row += f"{f'{sign}{delta:.4f}':>{col_w}}"

        print(row)


if __name__ == "__main__":
    main()
