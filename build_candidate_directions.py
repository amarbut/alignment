#!/usr/bin/env python3

"""
build_candidate_directions.py

Bridge script: load saved activations produced by activation_grab_v4.py
(harmless_activations.pkl, harmful_activations.pkl), compute per-(position,layer)
"refusal directions" as mean(harmful) - mean(harmless), and save a single
tensor with shape [n_pos, n_layer, d_model] that plugs directly into
select_direction.select_direction(...).

This is *not* production code—it's designed to be simple and explicit so you
can tweak it for your pipeline.

Output file is a Torch tensor saved via torch.save(...), defaulting to:
<activations_dir>/candidate_directions.pt

Assumptions
-----------
- Each pickled activation entry is a dict: layer_index -> Tensor[last_k, d_model]
  exactly as emitted by activation_grab_v4.py.
- All examples contain all layers; any missing entries are skipped with a warning.
- Position index 0 corresponds to token -last_k, and index last_k-1 to the last token.
  select_direction.py uses negative indices (range(-n_pos, 0)), which aligns
  with PyTorch negative indexing, so we keep positions in natural order and let
  the consumer index with negatives.

Formula
-------
For each layer ℓ and position p in {0..last_k-1}:
    dir[p, ℓ] = mean_over_examples( harmful[:, p, :] ) - mean_over_examples( harmless[:, p, :] )
Optionally we L2-normalize each direction vector (enabled by default).

Example
-------
python build_candidate_directions.py \
  --activations_dir /path/to/<model_name>/ \
  --save_path /path/to/<model_name>/candidate_directions.pt \
  --last_k 5 --no_normalize

After this, in your code you can do:
    candidate_directions = torch.load("/.../candidate_directions.pt")
    pos, layer, vec = select_direction(..., candidate_directions, ...)

"""
import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def _load_activation_list(pkl_path: Path) -> List[Dict[int, torch.Tensor]]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list from {pkl_path}, got {type(data)}")
    if len(data) == 0:
        raise ValueError(f"No entries in {pkl_path}")
    return data


def _stack_by_layer_and_pos(
    examples: List[Dict[int, torch.Tensor]], last_k: int
) -> Tuple[int, int, int, Dict[int, torch.Tensor]]:
    """
    Returns:
      n_examples, n_layers, d_model, stacked_per_layer
      where stacked_per_layer[layer] has shape [n_examples, last_k, d_model]
    """
    # Infer layers present from the first example
    first = examples[0]
    layer_ids = sorted(first.keys())
    n_layers = len(layer_ids)

    # Infer d_model from first layer tensor
    any_tensor = first[layer_ids[0]]
    if any_tensor.ndim != 2 or any_tensor.shape[0] != last_k:
        raise ValueError(
            f"Expected tensors shaped [last_k, d_model] with last_k={last_k}, "
            f"but got {tuple(any_tensor.shape)}"
        )
    d_model = any_tensor.shape[1]

    # Build stacks
    per_layer: Dict[int, List[torch.Tensor]] = {L: [] for L in layer_ids}
    n_examples_kept = 0
    for i, ex in enumerate(examples):
        missing = [L for L in layer_ids if L not in ex]
        if missing:
            # Skip examples that are missing any layer
            # (Alternatively, you could try to include partial data.)
            continue
        # Shape check & CPU float32
        ok = True
        for L in layer_ids:
            t = ex[L]
            if not (t.ndim == 2 and t.shape[0] == last_k):
                ok = False
                break
        if not ok:
            continue
        for L in layer_ids:
            per_layer[L].append(ex[L].detach().to("cpu", dtype=torch.float32))
        n_examples_kept += 1

    if n_examples_kept == 0:
        raise ValueError("No valid examples left after filtering.")

    stacked_per_layer: Dict[int, torch.Tensor] = {}
    for L in layer_ids:
        # -> [n_examples, last_k, d_model]
        stacked_per_layer[L] = torch.stack(per_layer[L], dim=0)

    return n_examples_kept, n_layers, d_model, stacked_per_layer


def _l2_normalize_rows(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    t: [..., d_model]
    Returns same shape, each row (last dim) normalized to unit norm.
    """
    norms = torch.linalg.norm(t, dim=-1, keepdim=True).clamp_min(eps)
    return t / norms


def build_candidate_directions(
    harmless_pkl: Path,
    harmful_pkl: Path,
    last_k: int,
    normalize: bool = True,
) -> torch.Tensor:
    harmless = _load_activation_list(harmless_pkl)
    harmful = _load_activation_list(harmful_pkl)

    n_harmless, n_layers_hl, d_model_hl, hl_by_layer = _stack_by_layer_and_pos(harmless, last_k)
    n_harmful,  n_layers_hf, d_model_hf, hf_by_layer = _stack_by_layer_and_pos(harmful,  last_k)

    if (n_layers_hl != n_layers_hf) or (d_model_hl != d_model_hf):
        raise ValueError(
            f"Shape mismatch between harmless (layers={n_layers_hl}, d={d_model_hl}) "
            f"and harmful (layers={n_layers_hf}, d={d_model_hf})"
        )

    layer_ids = sorted(hl_by_layer.keys())
    n_layers = len(layer_ids)
    d_model = d_model_hl

    # Output: [n_pos (=last_k), n_layer, d_model]
    candidate = torch.zeros((last_k, n_layers, d_model), dtype=torch.float32)

    for j, L in enumerate(layer_ids):
        # Shapes: [n_examples, last_k, d_model]
        hl_stack = hl_by_layer[L]  # harmless
        hf_stack = hf_by_layer[L]  # harmful

        # Means across examples -> [last_k, d_model]
        mu_hl = hl_stack.mean(dim=0)
        mu_hf = hf_stack.mean(dim=0)

        # Direction = harmful - harmless (aligns with "induce refusal" when added)
        dir_L = mu_hf - mu_hl  # [last_k, d_model]

        if normalize:
            dir_L = _l2_normalize_rows(dir_L)

        candidate[:, j, :] = dir_L  # position-major

    return candidate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations_dir", type=str, required=True,
                    help="Directory containing harmless_activations.pkl and harmful_activations.pkl")
    ap.add_argument("--last_k", type=int, default=5)
    ap.add_argument("--save_path", type=str, default=None,
                    help="Path to save candidate_directions.pt (default: <activations_dir>/candidate_directions.pt)")
    ap.add_argument("--no_normalize", action="store_true",
                    help="Disable L2-normalization of each direction vector")
    args = ap.parse_args()

    act_dir = Path(args.activations_dir)
    harmless_pkl = act_dir / "harmless_activations.pkl"
    harmful_pkl  = act_dir / "harmful_activations.pkl"

    if not harmless_pkl.exists() or not harmful_pkl.exists():
        raise FileNotFoundError(f"Could not find {harmless_pkl} and/or {harmful_pkl}")

    candidate = build_candidate_directions(
        harmless_pkl=harmless_pkl,
        harmful_pkl=harmful_pkl,
        last_k=args.last_k,
        normalize=(not args.no_normalize),
    )

    if args.save_path is None:
        save_path = act_dir / "candidate_directions.pt"
    else:
        save_path = Path(args.save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(candidate, save_path)

    print(f"[OK] Saved candidate_directions with shape {tuple(candidate.shape)} to: {save_path}")


if __name__ == "__main__":
    main()
