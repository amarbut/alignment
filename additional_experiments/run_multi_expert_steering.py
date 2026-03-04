"""
Multi-expert steering: combine the top-N single-expert results into joint interventions.

Algorithm
---------
1. Load an existing single-expert judge_grid_results.json (from a threshold run).
2. Extract the top N unique experts by KL-passing refusal score, one
   (position, coeff) setting per expert — the individually optimal setting
   already found by the single-expert pipeline.
3. Stage 1: For every C(N, K) combination apply all K hooks simultaneously,
            measure combined refusal score + KL divergence.
            Take the top M KL-passing combos by lowest refusal score.
4. Stage 2: Run the OpenAI judge on those top M combos; pick best ASR.
5. Full eval: Full-generation evaluation on the best combination.

Because positions and coefficients are fixed to each expert's individual
optimum, the only search dimension is which experts to combine — keeping
the combinatorial space at O(C(N, K)).

Usage
-----
  # Auto-locate single-expert results and run pairwise combinations:
  python run_multi_expert_steering.py \\
      --model_path allenai/OLMoE-1B-7B-0924-Instruct \\
      --system_prompt lightweight \\
      --top_n 10 --combo_k 2 \\
      --skip_harmless --skip_baseline --max_new_tokens 50

  # Point at a specific results file:
  python run_multi_expert_steering.py \\
      --model_path allenai/OLMoE-1B-7B-0924-Instruct \\
      --system_prompt lightweight \\
      --single_expert_results runs/.../judge_grid_results.json \\
      --top_n 8 --combo_k 3
"""

# =============================================================================
# Set HF cache BEFORE any HuggingFace imports
# =============================================================================
import sys
import argparse as _argparse_early

_early_parser = _argparse_early.ArgumentParser(add_help=False)
_early_parser.add_argument('--model_path', type=str, default='')
_early_args, _ = _early_parser.parse_known_args()
if _early_args.model_path:
    from model_utils.hf_cache_config import set_hf_cache_from_path
    set_hf_cache_from_path(_early_args.model_path)
# =============================================================================

try:
    from unsloth import FastLanguageModel
    from peft import PeftModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

import itertools
import json
import math
import os
import argparse

import torch
from tqdm import tqdm

from config import Config
from model_utils.model_factory_moe import construct_model_base
from submodules.arditi.select_direction import get_refusal_scores, get_last_position_logits, kl_div_fn
from submodules.evaluate_jailbreak import evaluate_jailbreak
from submodules.expert_steering.expert_intervention import (
    get_expert_weighted_activation_addition_hook,
    get_expert_weighted_intervention_hooks,
)
from run_expert_steering import (
    load_and_sample_datasets,
    filter_data,
    _get_shared_cache_dir,
    _load_or_compute_layer_directions,
)


# =============================================================================
# Expert extraction
# =============================================================================

def extract_top_n_experts(results, n, require_kl_pass=True):
    """
    From a list of single-expert stage-1/2 results, return the top N unique
    experts with their individually best (position, coeff) setting.

    "Best" = lowest refusal_score among KL-passing results (same ordering
    the single-expert pipeline used to build its stage-2 candidate list).

    Returns a list of dicts — each with keys:
        layer, expert_id, diff_pct, position, coeff, refusal_score
    sorted by refusal_score ascending (lower = stronger jailbreak signal).
    """
    # Deduplicate identical (layer, expert_id, position, coeff) entries
    seen = set()
    deduped = []
    for r in results:
        key = (r['layer'], r['expert_id'], r['position'], r['coeff'])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    if require_kl_pass:
        candidates = [r for r in deduped if r.get('passed_kl')]
    else:
        candidates = deduped

    # Keep the single best (position, coeff) per unique (layer, expert_id)
    by_expert = {}
    for r in candidates:
        key = (r['layer'], r['expert_id'])
        if key not in by_expert or r['refusal_score'] < by_expert[key]['refusal_score']:
            by_expert[key] = r

    sorted_experts = sorted(by_expert.values(), key=lambda x: x['refusal_score'])

    if n > len(sorted_experts):
        print(f"  Warning: only {len(sorted_experts)} unique KL-passing experts "
              f"available — using all of them.")
        n = len(sorted_experts)

    return sorted_experts[:n]


# =============================================================================
# Hook builders
# =============================================================================

def _combo_stage1_hooks(combo, model_card, device, dtype):
    """
    Build the fwd_hooks list for a stage-1 refusal-score evaluation of a combo.
    Each expert contributes one hook using the negative coeff convention
    (matching how run_grid_search handles stage-1 in the single-expert pipeline).
    """
    fwd_hooks = []
    for entry in combo:
        direction_vec = entry['direction'][entry['position']].to(device, dtype)
        hook_fn = get_expert_weighted_activation_addition_hook(
            direction=direction_vec,
            expert_id=entry['expert_id'],
            coeff=-entry['coeff'],  # negative: same sign convention as single-expert stage 1
            model_card=model_card,
        )
        mlp_module = model_card.get_mlp_module(entry['layer'])
        fwd_hooks.append((mlp_module, hook_fn))
    return fwd_hooks


def _combo_full_eval_hooks(combo, model_base, model_card, device, dtype):
    """
    Build (fwd_pre_hooks, fwd_hooks) for the full-generation evaluation of a
    combo, using the positive coeff convention (matching run_single_experiment).
    """
    all_pre, all_fwd = [], []
    for entry in combo:
        direction_vec = entry['direction'][entry['position']].to(device, dtype)
        pre, fwd = get_expert_weighted_intervention_hooks(
            model_base,
            layer_idx=entry['layer'],
            expert_id=entry['expert_id'],
            direction=direction_vec,
            coeff=entry['coeff'],   # positive: same sign convention as single-expert full eval
            model_card=model_card,
        )
        all_pre.extend(pre)
        all_fwd.extend(fwd)
    return all_pre, all_fwd


# =============================================================================
# Stage 1: refusal score + KL sweep over all C(N, K) combos
# =============================================================================

def run_multi_stage1(
    model_base, model_card, experts, combo_k,
    harmful_val_instructions, harmless_val_instructions,
    batch_size, kl_threshold,
):
    """
    Evaluate every C(N, K) combination of experts in stage-1 style:
    combined refusal score + KL divergence.

    Returns a list of result dicts, one per combo, sorted by refusal_score
    ascending (best first).
    """
    device = model_base.model.device
    dtype  = model_base.model.dtype

    all_combos = list(itertools.combinations(range(len(experts)), combo_k))
    n_combos = len(all_combos)

    print(f"\n  Combos: C({len(experts)}, {combo_k}) = {n_combos}")
    print(f"  KL threshold: {kl_threshold}")

    # Baseline (no intervention)
    print("  Computing baseline refusal scores and harmless logits...")
    baseline_scores = get_refusal_scores(
        model_base.model, harmful_val_instructions,
        model_base.tokenize_instructions_fn, model_base.refusal_toks,
        fwd_hooks=[], batch_size=batch_size,
        tokenizer=model_base.tokenizer,
        refusal_score_suffix_toks=model_base.refusal_score_suffix_toks,
    )
    baseline_mean = baseline_scores.mean().item()
    print(f"  Baseline refusal score: {baseline_mean:.4f}")

    baseline_harmless_logits = get_last_position_logits(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=harmless_val_instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        fwd_pre_hooks=[], fwd_hooks=[],
        batch_size=batch_size,
    )

    results = []
    with tqdm(all_combos, desc="  Stage 1 combos", unit="combo") as pbar:
        for idx_tuple in pbar:
            combo = [experts[i] for i in idx_tuple]
            fwd_hooks = _combo_stage1_hooks(combo, model_card, device, dtype)

            scores = get_refusal_scores(
                model_base.model, harmful_val_instructions,
                model_base.tokenize_instructions_fn, model_base.refusal_toks,
                fwd_hooks=fwd_hooks, batch_size=batch_size,
                tokenizer=model_base.tokenizer,
                refusal_score_suffix_toks=model_base.refusal_score_suffix_toks,
            )
            mean_score = scores.mean().item()

            intervention_logits = get_last_position_logits(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                instructions=harmless_val_instructions,
                tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                fwd_pre_hooks=[], fwd_hooks=fwd_hooks,
                batch_size=batch_size,
            )
            kl_div = kl_div_fn(
                baseline_harmless_logits, intervention_logits, mask=None
            ).mean(dim=0).item()
            passed_kl = kl_div <= kl_threshold

            results.append({
                "combo_indices": list(idx_tuple),
                "experts": [
                    {"layer": e['layer'], "expert_id": e['expert_id'],
                     "position": e['position'], "coeff": e['coeff'],
                     "diff_pct": e.get('diff_pct', 0.0)}
                    for e in combo
                ],
                "refusal_score": mean_score,
                "refusal_reduction": baseline_mean - mean_score,
                "kl_div": kl_div,
                "passed_kl": passed_kl,
            })

            best_so_far = min(
                (r['refusal_score'] for r in results if r['passed_kl']),
                default=float('inf')
            )
            pbar.set_postfix_str(
                f"score={mean_score:.3f} kl={'ok' if passed_kl else 'FAIL'} "
                f"best={best_so_far:.3f}"
            )

    results.sort(key=lambda x: x['refusal_score'])
    return results


# =============================================================================
# Stage 2: OpenAI judge on top-M combos
# =============================================================================

def run_multi_stage2(
    model_base, model_card, top_combos, experts,
    harmful_val_sample, max_new_tokens, batch_size, combos_dir,
):
    """
    Run the OpenAI judge on each of the top-M combos and return results
    with ASR added.
    """
    device = model_base.model.device
    dtype  = model_base.model.dtype
    best_asr = 0.0

    print(f"\n  Running OpenAI judge on {len(top_combos)} combos...")
    with tqdm(top_combos, desc="  Stage 2 judge", unit="combo") as pbar:
        for result in pbar:
            combo = [experts[i] for i in result['combo_indices']]
            pre, fwd = _combo_full_eval_hooks(combo, model_base, model_card, device, dtype)

            completions = model_base.generate_completions(
                harmful_val_sample,
                fwd_pre_hooks=pre,
                fwd_hooks=fwd,
                max_new_tokens=max_new_tokens,
            )

            label = "multi_" + "_".join(
                f"L{e['layer']}E{e['expert_id']}p{e['position']}c{e['coeff']}"
                for e in result['experts']
            )
            eval_path = os.path.join(combos_dir, f"{label}_eval.json") if combos_dir else os.devnull
            evaluation = evaluate_jailbreak(
                completions=completions,
                methodologies=["openai"],
                evaluation_path=eval_path,
                verbose=False,
            )

            asr = evaluation.get("openai_success_rate", 0.0)
            result['asr'] = asr
            result['full_response'] = evaluation.get("openai_overall_counts", {}).get("full_response", 0)
            result['refusal_count'] = evaluation.get("openai_overall_counts", {}).get("refusal", 0)
            result['non_response'] = evaluation.get("openai_overall_counts", {}).get("non_response", 0)

            best_asr = max(best_asr, asr)
            pbar.set_postfix_str(f"asr={asr:.0%}  best={best_asr:.0%}")

    return top_combos


# =============================================================================
# Full evaluation
# =============================================================================

def run_multi_full_eval(
    args, cfg, model_base, model_card,
    best_combo_result, experts,
    harmful_test, harmless_test,
    output_dir,
):
    """Full-generation evaluation on the best combination."""
    device = model_base.model.device
    dtype  = model_base.model.dtype

    combo = [experts[i] for i in best_combo_result['combo_indices']]
    label_parts = " + ".join(
        f"L{e['layer']} E{e['expert_id']} pos={e['position']} c={e['coeff']}"
        for e in best_combo_result['experts']
    )

    from dataset.load_dataset import load_dataset

    # Save experiment metadata
    metadata = {
        "combo": best_combo_result['experts'],
        "refusal_score": best_combo_result['refusal_score'],
        "kl_div": best_combo_result['kl_div'],
        "stage2_asr": best_combo_result.get('asr'),
        "selection_method": "multi_expert_judge_grid",
    }
    with open(os.path.join(output_dir, "experiment_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    pre, fwd = _combo_full_eval_hooks(combo, model_base, model_card, device, dtype)

    def _run_eval(dataset_name, eval_methodologies, dataset=None):
        completions_dir = os.path.join(output_dir, 'completions', 'actadd')
        os.makedirs(completions_dir, exist_ok=True)

        if dataset is None:
            import random
            dataset = load_dataset(dataset_name)
            dataset = random.sample(dataset, min(100, len(dataset)))

        print(f"\n  Generating completions for {dataset_name} [{label_parts}]...")
        completions = model_base.generate_completions(
            dataset,
            fwd_pre_hooks=pre,
            fwd_hooks=fwd,
            max_new_tokens=args.max_new_tokens,
        )
        completions_path = os.path.join(completions_dir, f"{dataset_name}_completions.json")
        with open(completions_path, 'w', encoding='utf-8') as f:
            json.dump(completions, f, indent=4, ensure_ascii=False)

        evaluation = evaluate_jailbreak(
            completions=completions,
            methodologies=eval_methodologies,
            evaluation_path=os.path.join(completions_dir, f"{dataset_name}_evaluations.json"),
        )
        with open(os.path.join(completions_dir, f"{dataset_name}_evaluations.json"), 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=4, ensure_ascii=False)

    # Baseline (optional)
    if not args.skip_baseline:
        print("\n  BASELINE (No Intervention)")
        for ds in cfg.evaluation_datasets:
            _run_eval_baseline(args, cfg, model_base, ds, output_dir)

    # Jailbreak eval
    for ds in cfg.evaluation_datasets:
        _run_eval(ds, cfg.jailbreak_eval_methodologies)

    # Harmless eval (optional)
    if not args.skip_harmless:
        _run_eval('harmless', cfg.refusal_eval_methodologies, dataset=harmless_test)


def _run_eval_baseline(args, cfg, model_base, dataset_name, output_dir):
    from dataset.load_dataset import load_dataset
    import random
    completions_dir = os.path.join(output_dir, 'completions', 'baseline')
    os.makedirs(completions_dir, exist_ok=True)
    dataset = load_dataset(dataset_name)
    dataset = random.sample(dataset, min(100, len(dataset)))
    completions = model_base.generate_completions(
        dataset, fwd_pre_hooks=[], fwd_hooks=[], max_new_tokens=args.max_new_tokens
    )
    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=cfg.jailbreak_eval_methodologies,
        evaluation_path=os.path.join(completions_dir, f"{dataset_name}_evaluations.json"),
    )
    with open(os.path.join(completions_dir, f"{dataset_name}_evaluations.json"), 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, indent=4, ensure_ascii=False)


# =============================================================================
# Argument parsing
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multi-expert steering: combine top-N single-expert results"
    )
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--system_prompt', type=str, default=None,
                        choices=['none', 'llama_2', 'lightweight'])

    # Multi-expert settings
    parser.add_argument('--top_n', type=int, default=10,
                        help='Top N unique experts from stage-1 to combine (default: 10)')
    parser.add_argument('--combo_k', type=int, default=2,
                        help='Number of experts per combination (default: 2)')
    parser.add_argument('--n_judge', type=int, default=None,
                        help='Combos to pass to stage-2 judge. Default: max(ceil(n_combos*0.1), 5)')

    # Source of single-expert stage-1 results
    parser.add_argument('--single_expert_results', type=str, default=None,
                        help='Path to judge_grid_results.json from a threshold run. '
                             'If omitted, auto-located under --runs_dir.')
    parser.add_argument('--runs_dir', type=str, default='runs',
                        help='Root runs directory for auto-locating results (default: runs/)')

    # Grid / judge
    parser.add_argument('--judge_grid_tokens', type=int, default=25,
                        help='Max tokens for stage-2 judge generations (default: 25)')
    parser.add_argument('--judge_grid_n_samples', type=int, default=25,
                        help='Harmful val samples for stage-2 judge (default: 25)')

    # Generation / evaluation
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--skip_baseline', action='store_true')
    parser.add_argument('--skip_harmless', action='store_true')
    parser.add_argument('--eval_datasets', type=str, nargs='+', default=None)
    parser.add_argument('--force_generate', action='store_true')

    # Dataset / sampling
    parser.add_argument('--n_train', type=int, default=None)
    parser.add_argument('--n_val',   type=int, default=None)
    parser.add_argument('--n_test',  type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_arguments()

    print("=" * 80)
    print("MULTI-EXPERT STEERING PIPELINE")
    print("=" * 80)
    print(f"Model:     {args.model_path}")
    print(f"Top N:     {args.top_n}")
    print(f"Combo K:   {args.combo_k}")
    print("=" * 80)

    base_model_name = os.path.basename(args.model_path)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    config_kwargs = {
        "model_alias": (f"{base_model_name}/expert_steering_multi{args.combo_k}"
                        f"/sys_prompt_{args.system_prompt}"),
        "model_path": args.model_path,
    }
    if args.system_prompt is not None:
        config_kwargs["system_prompt"] = args.system_prompt
    for k in ('n_train', 'n_val', 'n_test'):
        v = getattr(args, k)
        if v is not None:
            config_kwargs[k] = v
    cfg = Config(**config_kwargs)
    print(f"System prompt: {cfg.system_prompt}")

    if args.eval_datasets is not None:
        cfg.evaluation_datasets = tuple(args.eval_datasets)

    base_output_dir = os.path.join(cfg.artifact_path(), "grid_search")
    os.makedirs(base_output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Locate single-expert results
    # ------------------------------------------------------------------
    if args.single_expert_results:
        results_path = args.single_expert_results
    else:
        results_path = os.path.join(
            args.runs_dir, base_model_name,
            "expert_steering_threshold",
            f"sys_prompt_{cfg.system_prompt}",
            "grid_search", "judge_grid_search", "judge_grid_results.json",
        )
    if not os.path.exists(results_path):
        print(f"ERROR: single-expert results not found at:\n  {results_path}")
        print("Run the threshold mode first, or pass --single_expert_results <path>.")
        sys.exit(1)
    print(f"\nLoading single-expert stage-1 results from:\n  {results_path}")

    with open(results_path) as f:
        grid_data = json.load(f)

    # ------------------------------------------------------------------
    # Extract top-N experts
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXTRACTING TOP-N EXPERTS FROM SINGLE-EXPERT STAGE-1")
    print("=" * 80)
    experts_meta = extract_top_n_experts(
        grid_data['results'], n=args.top_n, require_kl_pass=True
    )
    print(f"\nTop {len(experts_meta)} unique experts (KL-passing, by refusal score):")
    print(f"  {'#':<5} {'Layer':<8} {'Expert':<8} {'Pos':<6} {'Coeff':<8} {'Refusal':>10}  {'Diff%':>8}")
    print("  " + "-" * 55)
    for i, e in enumerate(experts_meta, 1):
        print(f"  {i:<5} {e['layer']:<8} {e['expert_id']:<8} {e['position']:<6} "
              f"{e['coeff']:<8} {e['refusal_score']:>10.4f}  {e.get('diff_pct', 0):>8.2f}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    model_base = construct_model_base(args.model_path, system_prompt=cfg.system_prompt)

    from model_utils.model_card_factory import create_model_card
    model_card = create_model_card(model_base)
    kl_threshold = model_card.get_expert_steering_thresholds().get('kl_threshold', 1.0)

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)
    (harmful_train, harmless_train, harmful_val, harmless_val,
     harmful_test, harmless_test) = load_and_sample_datasets(cfg)
    print("\nFiltering training data based on refusal scores...")
    harmful_train, harmless_train = filter_data(cfg, model_base, harmful_train, harmless_train)

    # ------------------------------------------------------------------
    # Load directions from shared cache
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LOADING DIRECTIONS")
    print("=" * 80)
    shared_cache_dir = _get_shared_cache_dir(base_model_name, cfg.system_prompt)
    print(f"  Cache: {shared_cache_dir}/layer_{{L}}.pt")

    from collections import defaultdict
    layer_to_indices = defaultdict(list)
    for i, e in enumerate(experts_meta):
        layer_to_indices[e['layer']].append(i)

    for layer_idx in sorted(layer_to_indices):
        indices = layer_to_indices[layer_idx]
        expert_ids = [experts_meta[i]['expert_id'] for i in indices]
        layer_cache = _load_or_compute_layer_directions(
            layer_idx=layer_idx,
            expert_ids_needed=expert_ids,
            model_base=model_base,
            harmful_train=harmful_train,
            harmless_train=harmless_train,
            batch_size=args.batch_size,
            cache_dir=shared_cache_dir,
            force_generate=args.force_generate,
        )
        for i in indices:
            experts_meta[i]['direction'] = layer_cache[experts_meta[i]['expert_id']]

    # ------------------------------------------------------------------
    # Stage 1: sweep all C(N, K) combos
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"STAGE 1: REFUSAL SCORE + KL SWEEP  "
          f"(C({len(experts_meta)}, {args.combo_k}) = "
          f"{len(list(itertools.combinations(range(len(experts_meta)), args.combo_k)))} combos)")
    print("=" * 80)

    harmful_val_instructions = [
        x['instruction'] if isinstance(x, dict) else x for x in harmful_val
    ]
    harmless_val_instructions = [
        x['instruction'] if isinstance(x, dict) else x for x in harmless_val
    ]

    stage1_results = run_multi_stage1(
        model_base=model_base,
        model_card=model_card,
        experts=experts_meta,
        combo_k=args.combo_k,
        harmful_val_instructions=harmful_val_instructions,
        harmless_val_instructions=harmless_val_instructions,
        batch_size=args.batch_size,
        kl_threshold=kl_threshold,
    )

    # Save full stage-1 results
    stage1_dir = os.path.join(base_output_dir, "multi_judge_grid_search")
    os.makedirs(stage1_dir, exist_ok=True)
    with open(os.path.join(stage1_dir, "stage1_results.json"), 'w') as f:
        json.dump(stage1_results, f, indent=2)

    kl_passing = [r for r in stage1_results if r['passed_kl']]
    print(f"\n  Stage-1 done: {len(kl_passing)}/{len(stage1_results)} combos passed KL filter")

    if not kl_passing:
        print("ERROR: no combos passed the KL filter. "
              "Try a higher --kl_threshold or fewer experts.")
        sys.exit(1)

    # Determine how many combos to send to stage 2
    n_judge = args.n_judge if args.n_judge else max(math.ceil(len(kl_passing) * 0.1), 5)
    n_judge = min(n_judge, len(kl_passing))
    top_combos = kl_passing[:n_judge]

    print(f"  Sending top {n_judge} combos to stage-2 judge")
    print(f"  Top combos by refusal score:")
    for i, r in enumerate(top_combos[:5], 1):
        experts_str = " + ".join(
            f"L{e['layer']}E{e['expert_id']}(p{e['position']},c{e['coeff']})"
            for e in r['experts']
        )
        print(f"    {i}. {experts_str}  refusal={r['refusal_score']:.4f}")
    if len(top_combos) > 5:
        print(f"    ... and {len(top_combos) - 5} more")

    # ------------------------------------------------------------------
    # Stage 2: OpenAI judge on top combos
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 2: OPENAI JUDGE")
    print("=" * 80)

    combos_dir = os.path.join(stage1_dir, "combos")
    os.makedirs(combos_dir, exist_ok=True)

    harmful_val_sample = harmful_val[:args.judge_grid_n_samples]
    top_combos = run_multi_stage2(
        model_base=model_base,
        model_card=model_card,
        top_combos=top_combos,
        experts=experts_meta,
        harmful_val_sample=harmful_val_sample,
        max_new_tokens=args.judge_grid_tokens,
        batch_size=args.batch_size,
        combos_dir=combos_dir,
    )

    # Save stage-2 results
    with open(os.path.join(stage1_dir, "stage2_results.json"), 'w') as f:
        json.dump(top_combos, f, indent=2)

    best_combo = max(top_combos, key=lambda x: x.get('asr', 0))
    print(f"\n  Best combo ASR: {best_combo.get('asr', 0):.1%}")
    print(f"  Best combo experts:")
    for e in best_combo['experts']:
        print(f"    L{e['layer']} E{e['expert_id']}  pos={e['position']}  coeff={e['coeff']}  diff%={e.get('diff_pct',0):.2f}")

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FULL EVALUATION (best combination)")
    print("=" * 80)

    run_multi_full_eval(
        args=args,
        cfg=cfg,
        model_base=model_base,
        model_card=model_card,
        best_combo_result=best_combo,
        experts=experts_meta,
        harmful_test=harmful_test,
        harmless_test=harmless_test,
        output_dir=base_output_dir,
    )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Results saved under: {base_output_dir}")


if __name__ == "__main__":
    main()
