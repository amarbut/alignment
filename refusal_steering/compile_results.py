"""
Compile experiment results from run_olmoe_experiments.sh outputs.

Produces two sections:
  1. Main results table — per experiment per system prompt:
       - OpenAI judge ASR on jailbreakbench
       - Intervention parameters (coeff, threshold, layer/expert/position or forced/suppressed count)
       - Example completions sampled from full_response, refusal, non_response categories

  2. Multi-dataset ASR table — for Experiment 2 (threshold, full-length generation)
       across all five evaluation datasets

Usage:
  python compile_results.py
  python compile_results.py --runs_dir runs --model OLMoE-1B-7B-0924-Instruct
  python compile_results.py --output results_summary.txt
"""

import argparse
import json
import os
import textwrap
from glob import glob


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def asr_str(eval_data, dataset="jailbreakbench"):
    """Return 'XX.X%' OpenAI ASR string, or 'N/A' if missing."""
    if eval_data is None:
        return "N/A"
    rate = eval_data.get("openai_success_rate")
    if rate is None:
        return "N/A"
    return f"{rate:.1%}"


def counts_str(eval_data):
    """Return 'full=N refusal=N non_resp=N' from openai_overall_counts."""
    if eval_data is None:
        return "N/A"
    c = eval_data.get("openai_overall_counts", {})
    return (f"full={c.get('full_response', '?')}  "
            f"refusal={c.get('refusal', '?')}  "
            f"non_resp={c.get('non_response', '?')}")


def sample_completions(eval_data, n=2, max_chars=300):
    """
    Return a dict {category: [truncated completion strings]} with up to n
    examples each for 'full_response', 'refusal', 'non_response'.
    """
    if eval_data is None:
        return {}
    by_cat = {"full_response": [], "refusal": [], "non_response": []}
    for item in eval_data.get("completions", []):
        cat = item.get("openai_category")
        if cat in by_cat and len(by_cat[cat]) < n:
            prompt = item.get("prompt", "")[:80].strip()
            response = item.get("response", "").strip()
            if len(response) > max_chars:
                response = response[:max_chars] + "…"
            by_cat[cat].append((prompt, response))
    return by_cat


def format_completions(samples, indent="    "):
    """Format sampled completions into a readable block."""
    lines = []
    label_map = {
        "full_response": "FULL RESPONSE",
        "refusal": "REFUSAL",
        "non_response": "NON-RESPONSE",
    }
    for cat in ("full_response", "refusal", "non_response"):
        items = samples.get(cat, [])
        if not items:
            continue
        lines.append(f"{indent}[{label_map[cat]}]")
        for i, (prompt, response) in enumerate(items, 1):
            lines.append(f"{indent}  #{i} Prompt: {prompt}")
            wrapped = textwrap.fill(response, width=100,
                                    initial_indent=indent + "       ",
                                    subsequent_indent=indent + "       ")
            lines.append(f"{indent}     Response: {wrapped.strip()}")
        lines.append("")
    return "\n".join(lines)


def header(title, width=80):
    return "\n" + "=" * width + f"\n  {title}\n" + "=" * width


def subheader(title, width=70):
    return "\n" + "-" * width + f"\n  {title}\n" + "-" * width


# ---------------------------------------------------------------------------
# Experiment loaders
# ---------------------------------------------------------------------------

def load_exp1_allex(runs_dir, model):
    """Experiment 1: allex (expert_steering_allex)."""
    results = []
    base = os.path.join(runs_dir, model, "expert_steering_allex")
    for sp in ("none", "lightweight", "llama_2"):
        grid_dir = os.path.join(base, f"sys_prompt_{sp}", "grid_search")

        grid_results = load_json(os.path.join(grid_dir, "allex_grid_search", "allex_grid_results.json"))
        if grid_results is None:
            results.append({"sys_prompt": sp, "status": "missing"})
            continue

        best_layer = grid_results.get("best_layer", "?")
        best_pos   = grid_results.get("best_position", "?")
        best_coeff = grid_results.get("best_coeff", "?")

        # Find the final eval directory (allex_L{L}_pos{P}_c{C})
        eval_dir_pattern = os.path.join(grid_dir, f"allex_L{best_layer}_pos{best_pos}_c{best_coeff}")
        eval_path = os.path.join(eval_dir_pattern, "completions", "actadd",
                                 "jailbreakbench_evaluations.json")
        eval_data = load_json(eval_path)

        # Fallback: glob for any allex_L* eval dir if exact match is absent
        if eval_data is None:
            candidates = glob(os.path.join(grid_dir, "allex_L*", "completions", "actadd",
                                           "jailbreakbench_evaluations.json"))
            if candidates:
                eval_data = load_json(candidates[0])

        results.append({
            "sys_prompt": sp,
            "status": "ok" if eval_data else "eval_missing",
            "layer": best_layer,
            "position": best_pos,
            "coeff": best_coeff,
            "asr": asr_str(eval_data),
            "counts": counts_str(eval_data),
            "samples": sample_completions(eval_data),
        })
    return results


def load_exp2_threshold(runs_dir, model):
    """Experiment 2: threshold, multi-dataset (expert_steering_threshold)."""
    results = []
    datasets = ["jailbreakbench", "advbench", "tdc2023", "maliciousinstruct", "strongreject"]
    base = os.path.join(runs_dir, model, "expert_steering_threshold")

    for sp in ("none", "lightweight", "llama_2"):
        grid_dir = os.path.join(base, f"sys_prompt_{sp}", "grid_search")
        meta = load_json(os.path.join(grid_dir, "experiment_metadata.json"))

        dataset_evals = {}
        for ds in datasets:
            eval_path = os.path.join(grid_dir, "completions", "actadd", f"{ds}_evaluations.json")
            dataset_evals[ds] = load_json(eval_path)

        jbb_eval = dataset_evals.get("jailbreakbench")

        entry = {
            "sys_prompt": sp,
            "status": "ok" if meta else "missing",
            "meta": meta or {},
            "dataset_evals": dataset_evals,
            "datasets": datasets,
            "asr": asr_str(jbb_eval),
            "counts": counts_str(jbb_eval),
            "samples": sample_completions(jbb_eval),
        }
        results.append(entry)
    return results


def load_exp3_arditi(runs_dir, model):
    """Experiment 3: Arditi pipeline."""
    results = []
    base = os.path.join(runs_dir, model, "arditi")

    for sp in ("none", "lightweight", "llama_2"):
        sp_dir = os.path.join(base, f"sys_prompt_{sp}")
        meta = load_json(os.path.join(sp_dir, "metadata.json"))

        eval_path = os.path.join(sp_dir, "completions", "jailbreakbench_actadd_evaluations.json")
        eval_data = load_json(eval_path)

        entry = {
            "sys_prompt": sp,
            "status": "ok" if eval_data else "missing",
            "meta": meta or {},
            "asr": asr_str(eval_data),
            "counts": counts_str(eval_data),
            "samples": sample_completions(eval_data),
        }
        results.append(entry)
    return results


def load_exp4_routing(runs_dir, model, threshold=15.0):
    """Experiment 4: expert routing (select_experts intervention)."""
    results = []
    base = os.path.join(runs_dir, model, f"expert_routing_t{threshold}")

    for sp in ("none", "lightweight", "llama_2"):
        sp_dir = os.path.join(base, f"sys_prompt_{sp}")
        # select_refusal = suppress harmful-preferring experts → jailbreak attack
        refusal_dir = os.path.join(sp_dir, "completions", "select_refusal")
        meta   = load_json(os.path.join(refusal_dir, "metadata.json"))
        eval_data = load_json(os.path.join(refusal_dir, "jailbreakbench_evaluations.json"))

        entry = {
            "sys_prompt": sp,
            "status": "ok" if eval_data else "missing",
            "meta": meta or {},
            "asr": asr_str(eval_data),
            "counts": counts_str(eval_data),
            "samples": sample_completions(eval_data),
        }
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def fmt_exp1(results):
    lines = [header("EXPERIMENT 1 — Allex Expert Steering (all-expert mean-diff, grid search)")]
    for r in results:
        lines.append(subheader(f"System prompt: {r['sys_prompt']}"))
        if r["status"] == "missing":
            lines.append("  [NO RESULTS FOUND]")
            continue

        lines.append(f"  Layer:    {r['layer']}")
        lines.append(f"  Position: {r['position']}")
        lines.append(f"  Coeff:    {r['coeff']}")
        lines.append(f"  ASR (JBB, OpenAI): {r['asr']}   ({r['counts']})")
        lines.append("")
        lines.append("  Example completions:")
        lines.append(format_completions(r["samples"], indent="    "))
    return "\n".join(lines)


def fmt_exp2(results):
    lines = [header("EXPERIMENT 2 — Threshold Expert Steering (multi-dataset, full generation)")]

    for r in results:
        lines.append(subheader(f"System prompt: {r['sys_prompt']}"))
        if r["status"] == "missing":
            lines.append("  [NO RESULTS FOUND]")
            continue

        m = r["meta"]
        lines.append(f"  Layer:     {m.get('layer', '?')}")
        lines.append(f"  Expert:    {m.get('expert', '?')}")
        lines.append(f"  Diff%:     {m.get('diff_pct', '?'):.2f}%" if isinstance(m.get('diff_pct'), float) else f"  Diff%:     {m.get('diff_pct', '?')}")
        lines.append(f"  Rank:      {m.get('rank', '?')}")
        lines.append(f"  Position:  {m.get('position', '?')}")
        lines.append(f"  Coeff:     {m.get('coeff', '?')}")
        lines.append(f"  ASR (JBB, OpenAI): {r['asr']}   ({r['counts']})")
        lines.append("")
        lines.append("  Example completions (jailbreakbench):")
        lines.append(format_completions(r["samples"], indent="    "))

    # Multi-dataset table
    lines.append(subheader("Multi-dataset ASR table (OpenAI judge)"))
    datasets = results[0]["datasets"] if results else []
    col_w = 20
    header_row = f"  {'sys_prompt':<15}" + "".join(f"{ds:>{col_w}}" for ds in datasets)
    lines.append(header_row)
    lines.append("  " + "-" * (15 + col_w * len(datasets)))
    for r in results:
        row = f"  {r['sys_prompt']:<15}"
        for ds in datasets:
            ev = r["dataset_evals"].get(ds)
            row += f"{asr_str(ev):>{col_w}}"
        lines.append(row)

    return "\n".join(lines)


def fmt_exp3(results):
    lines = [header("EXPERIMENT 3 — Arditi Refusal Direction Steering")]
    for r in results:
        lines.append(subheader(f"System prompt: {r['sys_prompt']}"))
        if r["status"] == "missing":
            lines.append("  [NO RESULTS FOUND]")
            continue

        m = r["meta"]
        lines.append(f"  Layer:    {m.get('layer', '?')}")
        lines.append(f"  Position: {m.get('position', '?')}")
        lines.append(f"  Coeff:    {m.get('coefficient', '?')}")
        lines.append(f"  ASR (JBB, OpenAI): {r['asr']}   ({r['counts']})")
        lines.append("")
        lines.append("  Example completions:")
        lines.append(format_completions(r["samples"], indent="    "))
    return "\n".join(lines)


def fmt_exp4(results, threshold):
    lines = [header(f"EXPERIMENT 4 — Expert Routing (threshold={threshold}, select_experts → suppress harmful)")]
    for r in results:
        lines.append(subheader(f"System prompt: {r['sys_prompt']}"))
        if r["status"] == "missing":
            lines.append("  [NO RESULTS FOUND]")
            continue

        m = r["meta"]
        lines.append(f"  Threshold:          {m.get('threshold', threshold)}")
        lines.append(f"  Suppressed experts: {m.get('num_suppressed_experts', '?')}")
        lines.append(f"  Forced experts:     {m.get('num_forced_experts', '?')}")
        lines.append(f"  ASR (JBB, OpenAI):  {r['asr']}   ({r['counts']})")
        lines.append("")
        lines.append("  Example completions:")
        lines.append(format_completions(r["samples"], indent="    "))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compile OLMoE experiment results")
    parser.add_argument("--runs_dir", default="runs",
                        help="Root runs directory (default: runs/)")
    parser.add_argument("--model", default="OLMoE-1B-7B-0924-Instruct",
                        help="Model subdirectory name (default: OLMoE-1B-7B-0924-Instruct)")
    parser.add_argument("--routing_threshold", type=float, default=15.0,
                        help="Routing threshold used in Experiment 4 (default: 15.0)")
    parser.add_argument("--output", default=None,
                        help="Write output to this file in addition to stdout")
    args = parser.parse_args()

    sections = []

    print(f"Loading Experiment 1 (allex)...")
    sections.append(fmt_exp1(load_exp1_allex(args.runs_dir, args.model)))

    print(f"Loading Experiment 2 (threshold, multi-dataset)...")
    sections.append(fmt_exp2(load_exp2_threshold(args.runs_dir, args.model)))

    print(f"Loading Experiment 3 (Arditi)...")
    sections.append(fmt_exp3(load_exp3_arditi(args.runs_dir, args.model)))

    print(f"Loading Experiment 4 (expert routing)...")
    sections.append(fmt_exp4(load_exp4_routing(args.runs_dir, args.model, args.routing_threshold),
                             args.routing_threshold))

    output = "\n\n".join(sections) + "\n"

    print("\n" + output)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()
