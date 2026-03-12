#!/usr/bin/env python3
"""
Package analysis artifacts for local transfer.

Collects:
  - Arditi evaluations + extracted direction vectors for each (layer, pos) with completions
  - Expert directions (fp16) for only the layers needed by Arditi results
  - Expert steering grid result JSONs and final evaluation JSONs
  - Expert diffs JSONs

Configure the variables in the CONFIG section, then run:
    python package_for_transfer.py
"""

# =============================================================================
# CONFIG — edit these for each model/instance
# =============================================================================

REFUSAL_STEERING_DIR = "/media/volume/oss20B_olmoe/alignment/refusal_steering"

MODEL_DIR_NAME  = "OLMoE-1B-7B-0924-Instruct"   # subfolder under runs/
MODEL_SHORTNAME = "olmoe"                          # prefix in expert_diffs filenames
SYS_PROMPTS     = ["none", "lightweight", "llama_2"]

HAS_ARDITI = True   # set False if no arditi/ run dir for this model

OUTPUT_DIR = f"/tmp/transfer_{MODEL_SHORTNAME}"
TAR_PATH   = f"{REFUSAL_STEERING_DIR}/{MODEL_SHORTNAME}_analysis_package.tar.gz"

# =============================================================================
import os, re, shutil, glob, tarfile, json
from pathlib import Path
import torch

RUNS_MODEL_DIR = os.path.join(REFUSAL_STEERING_DIR, "runs", MODEL_DIR_NAME)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mirror_copy(src: str, dst_base: str, src_base: str):
    rel = os.path.relpath(src, src_base)
    dst = os.path.join(dst_base, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  copy  {rel}")
    return dst


def save_tensor(tensor, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(tensor, dst)


def convert_pt_fp16(src: str, dst_base: str, src_base: str):
    rel = os.path.relpath(src, src_base)
    dst = os.path.join(dst_base, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    data = torch.load(src, map_location="cpu")
    if isinstance(data, torch.Tensor):
        data = data.to(torch.float16)
    elif isinstance(data, dict):
        data = {
            k: v.to(torch.float16) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }

    torch.save(data, dst)
    src_mb = os.path.getsize(src) / 1024**2
    dst_mb = os.path.getsize(dst) / 1024**2
    print(f"  fp16  {rel}  ({src_mb:.1f} MB → {dst_mb:.1f} MB)")
    return dst


def glob_files(pattern: str):
    return sorted(glob.glob(pattern, recursive=True))


# ---------------------------------------------------------------------------
# Parse layer/pos from completions filenames
# e.g. "jailbreakbench_actadd_L9_P-1_evaluations.json" → (9, -1)
#      "jailbreakbench_actadd_evaluations.json"         → uses direction_metadata.json
# ---------------------------------------------------------------------------

def parse_layer_pos_from_filename(fname: str):
    """Return (layer, pos) if encoded in filename, else None."""
    m = re.search(r'_L(\d+)_P(-?\d+)', fname)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def get_arditi_layer_pos_sets() -> dict:
    """
    For each sys_prompt, return the set of (layer, pos) combinations that
    have arditi evaluation files, reading direction_metadata.json as fallback
    for filenames without an explicit L/P encoding.

    Returns: {sys_prompt: set of (layer, pos)}
    """
    result = {}
    if not HAS_ARDITI:
        return result

    arditi_base = os.path.join(RUNS_MODEL_DIR, "arditi")
    for sp in SYS_PROMPTS:
        sp_dir = os.path.join(arditi_base, f"sys_prompt_{sp}")
        if not os.path.isdir(sp_dir):
            continue

        # Default from direction_metadata.json
        meta_path = os.path.join(sp_dir, "direction_metadata.json")
        default_lp = None
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            # pos in metadata is stored as the position index (0-4) or as -5 to -1
            raw_pos = meta["pos"]
            pos = raw_pos if raw_pos < 0 else raw_pos - 5   # normalise to -5..-1
            default_lp = (meta["layer"], pos)

        lp_set = set()
        for eval_file in glob_files(f"{sp_dir}/completions/*_evaluations.json"):
            fname = os.path.basename(eval_file)
            lp = parse_layer_pos_from_filename(fname)
            if lp:
                lp_set.add(lp)
            elif default_lp:
                lp_set.add(default_lp)

        if lp_set:
            result[sp] = lp_set
            print(f"  sys_prompt_{sp}: arditi (layer, pos) = {sorted(lp_set)}")

    return result


# ---------------------------------------------------------------------------
# Collection functions
# ---------------------------------------------------------------------------

def collect_arditi(output_dir: str, arditi_lp_sets: dict):
    if not HAS_ARDITI:
        return
    print("\n[Arditi]")
    arditi_base = os.path.join(RUNS_MODEL_DIR, "arditi")
    if not os.path.isdir(arditi_base):
        print("  !! arditi/ directory not found, skipping")
        return

    for sp in SYS_PROMPTS:
        sp_dir = os.path.join(arditi_base, f"sys_prompt_{sp}")
        if not os.path.isdir(sp_dir):
            continue

        # direction_metadata.json
        meta_path = os.path.join(sp_dir, "direction_metadata.json")
        if os.path.exists(meta_path):
            mirror_copy(meta_path, output_dir, REFUSAL_STEERING_DIR)

        # evaluation JSONs
        for f in glob_files(f"{sp_dir}/completions/*_evaluations.json"):
            mirror_copy(f, output_dir, REFUSAL_STEERING_DIR)

        # Extract specific direction vectors from mean_diffs.pt
        mean_diffs_path = os.path.join(sp_dir, "generate_directions", "mean_diffs.pt")
        lp_set = arditi_lp_sets.get(sp, set())
        if os.path.exists(mean_diffs_path) and lp_set:
            mean_diffs = torch.load(mean_diffs_path, map_location="cpu")
            # mean_diffs shape: [n_positions, n_layers, d_model]
            # positions index: 0=-5, 1=-4, 2=-3, 3=-2, 4=-1
            directions = {}
            for (layer, pos) in sorted(lp_set):
                pos_idx = pos + 5   # -5→0, -4→1, ..., -1→4
                d = mean_diffs[pos_idx, layer, :].to(torch.float16)
                key = f"L{layer}_P{pos}"
                directions[key] = d
                print(f"  extract  arditi direction {key}  shape={tuple(d.shape)}")

            dst_path = os.path.join(
                output_dir,
                os.path.relpath(sp_dir, REFUSAL_STEERING_DIR),
                "arditi_directions.pt"
            )
            save_tensor(directions, dst_path)
            print(f"  saved  {os.path.relpath(dst_path, output_dir)}")


def collect_expert_directions(output_dir: str, arditi_lp_sets: dict):
    """
    Collect expert directions only for layers that appear in Arditi results.
    For each Arditi layer L we collect:
      - expert_directions at layer L   (for projection analysis: compare experts vs arditi)
      - expert_directions at layer L-1 (for matched-location analysis: expert steering L-1 ≈ arditi L)
    """
    print("\n[Expert directions]")
    ed_base = os.path.join(RUNS_MODEL_DIR, "expert_directions")
    if not os.path.isdir(ed_base):
        print("  !! expert_directions/ not found, skipping")
        return

    # Collect union of needed layers across all sys_prompts
    needed_layers = set()
    for lp_set in arditi_lp_sets.values():
        for (layer, _) in lp_set:
            needed_layers.add(layer)       # same layer for projection
            if layer > 0:
                needed_layers.add(layer - 1)   # L-1 for matched-location

    print(f"  Needed layers: {sorted(needed_layers)}")

    for sp in SYS_PROMPTS:
        sp_dir = os.path.join(ed_base, f"sys_prompt_{sp}")
        if not os.path.isdir(sp_dir):
            continue
        for layer in sorted(needed_layers):
            pt_file = os.path.join(sp_dir, f"layer_{layer}.pt")
            if os.path.exists(pt_file):
                convert_pt_fp16(pt_file, output_dir, REFUSAL_STEERING_DIR)
            else:
                print(f"  !! missing  {os.path.relpath(pt_file, REFUSAL_STEERING_DIR)}")


def collect_expert_steering(output_dir: str):
    """
    Collect grid result summaries and final evaluation JSONs for all
    expert_steering_* runs. Skips ablation, benchmark, and routing runs.
    """
    print("\n[Expert steering results]")

    # Summary JSONs we want
    summary_names = {
        "judge_grid_results.json",
        "allex_grid_results.json",
        "experiment_metadata.json",
    }
    # Final evaluation dirs we want (skip select_refusal / select_response)
    eval_subdirs = {"actadd", "baseline"}

    seen = set()

    for run_dir in sorted(glob.glob(os.path.join(RUNS_MODEL_DIR, "expert_steering_*"))):
        run_name = os.path.basename(run_dir)
        # Skip ablation runs (named like expert_steering_ablation)
        if "ablation" in run_name:
            continue

        for root, dirs, files in os.walk(run_dir):
            # Skip combos/ subdirs — they're per-combination trial files
            dirs[:] = [d for d in dirs if d != "combos"]

            for fname in files:
                fpath = os.path.join(root, fname)

                # Grid summary JSONs
                if fname in summary_names and fpath not in seen:
                    seen.add(fpath)
                    mirror_copy(fpath, output_dir, REFUSAL_STEERING_DIR)

                # Evaluation JSONs from completions/actadd/ and completions/baseline/
                if fname.endswith("_evaluations.json") and fpath not in seen:
                    parts = Path(root).parts
                    if len(parts) >= 2 and parts[-1] in eval_subdirs and parts[-2] == "completions":
                        seen.add(fpath)
                        mirror_copy(fpath, output_dir, REFUSAL_STEERING_DIR)


def collect_expert_diffs(output_dir: str):
    print("\n[Expert diffs]")
    for diffs_dir in ["expert_diffs", "expert_diffs_topk"]:
        base = os.path.join(REFUSAL_STEERING_DIR, diffs_dir)
        if not os.path.isdir(base):
            continue
        # Sys-prompt subdirectories
        for sp in SYS_PROMPTS:
            sp_dir = os.path.join(base, f"sys_prompt_{sp}")
            for f in glob_files(f"{sp_dir}/{MODEL_SHORTNAME}_*.json"):
                mirror_copy(f, output_dir, REFUSAL_STEERING_DIR)
        # Root-level (no sys_prompt subfolder, if any)
        for f in glob_files(f"{base}/{MODEL_SHORTNAME}_*.json"):
            mirror_copy(f, output_dir, REFUSAL_STEERING_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Model:  {MODEL_DIR_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Tar:    {TAR_PATH}\n")

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print("[Scanning Arditi layer/pos combinations]")
    arditi_lp_sets = get_arditi_layer_pos_sets()

    collect_arditi(output_dir=OUTPUT_DIR, arditi_lp_sets=arditi_lp_sets)
    collect_expert_directions(output_dir=OUTPUT_DIR, arditi_lp_sets=arditi_lp_sets)
    collect_expert_steering(output_dir=OUTPUT_DIR)
    collect_expert_diffs(output_dir=OUTPUT_DIR)

    print(f"\n[Tar] Creating {TAR_PATH} ...")
    with tarfile.open(TAR_PATH, "w:gz") as tar:
        tar.add(OUTPUT_DIR, arcname=MODEL_SHORTNAME)

    tar_mb = os.path.getsize(TAR_PATH) / 1024**2
    output_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(OUTPUT_DIR)
        for f in files
    ) / 1024**2
    print(f"Done.")
    print(f"  Uncompressed: {output_mb:.1f} MB")
    print(f"  Tar size:     {tar_mb:.1f} MB")
    print(f"\nTransfer with:")
    print(f"  scp user@host:{TAR_PATH} .")
    print(f"  tar xzf {os.path.basename(TAR_PATH)}")


if __name__ == "__main__":
    main()
