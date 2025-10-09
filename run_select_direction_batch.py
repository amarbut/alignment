#!/usr/bin/env python3

"""
run_select_direction_batch.py

Batch runner that:
  1) Loads harmful/harmless instructions from files,
  2) Loads candidate_directions.pt produced by build_candidate_directions.py,
  3) Instantiates a ModelBase (via a user-specified loader),
  4) Calls select_direction.select_direction(...) to pick a single direction,
  5) Saves the chosen (pos, layer, direction) per model.

This is a thin orchestrator to plug your activations into Arditi's pipeline.

USAGE (example)
--------------
python run_select_direction_batch.py \
  --config /path/to/config.json \
  --loader-func pipeline.model_utils.loaders:load_model_base \
  --kl-threshold 0.1 \
  --induce-threshold 0.0 \
  --prune-layer-percentage 0.2 \
  --batch-size 32

CONFIG FORMAT (JSON)
--------------------
{
  "prompts": {
    "harmful_file": "/data/prompts/harmful.jsonl",
    "harmless_file": "/data/prompts/harmless.txt"
  },
  "models": [
    {
      "name": "llama3.1-8b",
      "activations_dir": "/exp/acts/llama3.1-8b",
      "artifact_dir": "/exp/artifacts/llama3.1-8b",
      "loader_kwargs": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B",
        "device": "cuda:0",
        "dtype": "bfloat16"
      }
    },
    {
      "name": "mixtral-8x7b",
      "activations_dir": "/exp/acts/mixtral-8x7b",
      "artifact_dir": "/exp/artifacts/mixtral-8x7b",
      "loader_kwargs": {
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "device": "cuda:0",
        "dtype": "bfloat16"
      }
    }
  ]
}

LOADER
------
You must provide a function via --loader-func in the form "module.sub:callable".
The callable must return an instance of ModelBase when invoked with **loader_kwargs.
For example, you can implement pipeline.model_utils.loaders.load_model_base(**kwargs)
in your codebase that constructs the right ModelBase subclass.

OUTPUTS
-------
For each model, we write into <artifact_dir>:
  - selected_direction.pt     (Tensor[d_model])
  - selected_meta.json        {"name", "pos", "layer", ...}
We also echo a short summary to stdout.

NOTE
----
This script assumes your environment has the same dependencies used by
select_direction.py (torch, jaxtyping, einops, etc.) and that your
--loader-func returns a ready ModelBase (with tokenizer, refusal tokens, etc.).
"""

# add arditi direction to path to simplify imports
import sys
from pathlib import Path

# Adjust this to where you nested Arditi's repo
ARDITI_ROOT = Path(__file__).resolve().parents[1] / "refusal_direction"
sys.path.insert(0, str(ARDITI_ROOT))

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

# Import the selector from your attached file
from pipeline.submodules.select_direction import select_direction  # noqa


# ---------------------------
# Utilities
# ---------------------------

def _import_callable(path: str) -> Callable:
    """
    Import a Python callable from a 'module.sub:callable' string.
    """
    if ":" not in path:
        raise ValueError("loader-func must be in the form 'module.sub:callable'")
    mod_name, attr = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise TypeError(f"{path} is not callable")
    return fn


def _read_prompts(file_path: str) -> List[str]:

    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {file_path}")

    with open(p, "r") as f:
        dataset = [d["instruction"] for d in json.load(f)]
    
    return dataset


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="Path to JSON config (see header).")
    ap.add_argument("--loader_func", required=True, type=str, help="Import path 'module:callable' returning a ModelBase.")
    ap.add_argument("--kl_threshold", type=float, default=0.1)
    ap.add_argument("--induce_threshold", type=float, default=0.0)
    ap.add_argument("--prune_layer_percentage", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    print(f"[debug] batch_size at runner={args.batch_size}")
    
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())

    harmful_file = cfg["prompts"]["harmful_file"]
    harmless_file = cfg["prompts"]["harmless_file"]

    harmful_prompts = _read_prompts(harmful_file)
    harmless_prompts = _read_prompts(harmless_file)

    load_model_base = _import_callable(args.loader_func)

    for m in cfg["models"]:
        name = m["name"]
        acts_dir = Path(m["activations_dir"])
        art_dir = Path(m["artifact_dir"])
        art_dir.mkdir(parents=True, exist_ok=True)

        candidate_path = acts_dir / "candidate_directions.pt"
        if not candidate_path.exists():
            raise FileNotFoundError(f"[{name}] Missing candidate_directions.pt at {candidate_path}")

        candidate_directions = torch.load(candidate_path, map_location="cpu")
        if not isinstance(candidate_directions, torch.Tensor) or candidate_directions.ndim != 3:
            raise ValueError(f"[{name}] candidate_directions must be a 3D tensor [n_pos, n_layer, d_model]")

        print(f"\n=== Model: {name} ===")
        print(f"- Loading ModelBase with kwargs: {m.get('loader_kwargs', {})}")
        model_base = load_model_base(**m.get("loader_kwargs", {}))

        # Move candidate to model device for speed
        candidate_directions = candidate_directions.to(model_base.model.device)

        pos, layer, direction = select_direction(
            model_base=model_base,
            harmful_instructions=harmful_prompts,
            harmless_instructions=harmless_prompts,
            candidate_directions=candidate_directions,
            artifact_dir=str(art_dir),
            kl_threshold=args.kl_threshold,
            induce_refusal_threshold=args.induce_threshold,
            prune_layer_percentage=args.prune_layer_percentage,
            batch_size=args.batch_size,
        )

        # Save artifacts
        torch.save(direction.detach().to("cpu"), art_dir / "selected_direction.pt")
        meta = {
            "name": name,
            "pos": int(pos),
            "layer": int(layer),
            "kl_threshold": args.kl_threshold,
            "induce_refusal_threshold": args.induce_threshold,
            "prune_layer_percentage": args.prune_layer_percentage,
            "batch_size": args.batch_size,
            "candidate_shape": list(candidate_directions.shape),
        }
        (art_dir / "selected_meta.json").write_text(json.dumps(meta, indent=2))

        print(f"[OK] Saved selection for {name}:")
        print(f"  - pos={pos}, layer={layer}")
        print(f"  - direction -> {art_dir / 'selected_direction.pt'}")
        print(f"  - meta      -> {art_dir / 'selected_meta.json'}")


if __name__ == "__main__":
    main()
