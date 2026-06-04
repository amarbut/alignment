# Expert-Aware Refusal Steering

Code for the paper:

> **Expert-Aware Refusal Steering**  
> Anna C. Marbut, Travis J. Wheeler, Daniel R. Olson  
> *Preprint under review, Conference on Language Modeling (COLM 2026)*  
> [Preprint PDF](https://amarbut.github.io/files/MoE_Steering_COLM_preprint.pdf)

## Overview

Safety alignment in instruction-tuned LLMs depends on reliable refusal behavior. Recent work has shown that steering vectors applied during inference can suppress refusal in dense models. We extend this to Mixture-of-Experts (MoE) architectures and investigate how MoE routing interacts with refusal steering.

**Key findings:**
- Refusal steering transfers directly from dense to MoE models — routing complexity does not prevent effective steering
- Refusal behavior is distributed across **both attention and feed-forward sublayers**, not concentrated in specialized MoE experts
- Expert routing patterns don't predict steering effectiveness (detection ≠ response)
- Evidence of **two distinct refusal pathways**: FFN-mediated (internal) and attention-mediated (contextual)
- Post-trained models show **behavioral entanglement** between refusal and adjacent behavioral dimensions, consistent with superposition

We evaluate on three open-source MoE models: [Mixtral-8x7B-Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1), [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct), and [DeepSeek-MoE](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat).

## Repository Structure

```
config.py                   Experiment configuration dataclass
run_expert_steering.py      Expert-specific steering experiments (6 modes; see docstring)
run_expert_routing.py       Expert routing pattern analysis
run_pipeline_arditi.py      Baseline Arditi (dense) steering pipeline

model_utils/                Model loading and MoE model cards for each supported architecture
submodules/
  arditi/                   Arditi refusal direction generation and hook utilities
  expert_steering/          Expert-specific steering: direction selection and intervention
  expert_routing/           Expert routing hooks and analysis
  evaluate_jailbreak.py     Jailbreak evaluation (substring matching + OpenAI judge)
  expert_diff_generator.py  Generate expert routing difference vectors

additional_experiments/     Supporting experiments
  run_benchmarks.py         Run standard safety benchmarks
  run_ablation_random_experts.py  Random expert ablation study
  run_expert_diffs.py       Expert difference analysis
  run_multi_expert_steering.py  Multi-expert steering experiments

compile_results.py          Aggregate and display results across runs

dataset/
  raw/                      Original benchmark data (AdvBench, HarmBench, JailbreakBench, etc.)
  processed/                Pre-processed evaluation sets
  load_dataset.py           Dataset loading utilities
```

## Supported Models

| Model | HuggingFace Path |
|---|---|
| Mixtral-8x7B-Instruct | `mistralai/Mixtral-8x7B-Instruct-v0.1` |
| OLMoE-1B-7B-Instruct | `allenai/OLMoE-1B-7B-0924-Instruct` |
| DeepSeek-MoE | `deepseek-ai/deepseek-moe-16b-chat` |

## Quickstart

**Generate refusal steering directions (Arditi method):**
```bash
python -m submodules.arditi.generate_directions --model_path allenai/OLMoE-1B-7B-0924-Instruct
```

**Run expert-specific steering:**
```bash
# Default: threshold-based expert selection, grid search over coefficients
python run_expert_steering.py --model_path allenai/OLMoE-1B-7B-0924-Instruct

# All experts at all layers
python run_expert_steering.py --model_path allenai/OLMoE-1B-7B-0924-Instruct --allex

# Top-N experts by routing difference
python run_expert_steering.py --model_path allenai/OLMoE-1B-7B-0924-Instruct --expert_rank 5
```

**Run baseline Arditi pipeline:**
```bash
python run_pipeline_arditi.py --model_path allenai/OLMoE-1B-7B-0924-Instruct
```

**Analyze expert routing patterns:**
```bash
python run_expert_routing.py --model_path allenai/OLMoE-1B-7B-0924-Instruct
```

Experiment outputs are written to `runs/<model_alias>/`.

## Configuration

Key parameters are defined in `config.py`. The most commonly varied fields:

| Parameter | Default | Description |
|---|---|---|
| `n_train` | 128 | Training set size for direction extraction |
| `n_test` | 100 | Evaluation set size |
| `coeff` | 1.0 | Steering coefficient |
| `threshold` | 15.0 | Expert selection threshold |
| `system_prompt` | `"lightweight"` | System prompt style: `"none"`, `"lightweight"`, `"llama_2"` |
| `evaluation_datasets` | `("jailbreakbench",)` | Datasets to evaluate on |

## Requirements

```
torch
transformers
datasets
numpy
pandas
```

For OpenAI-based jailbreak evaluation: `openai` (requires API key in `OPENAI_API_KEY`).

## Citation

```bibtex
@article{marbut2026expert,
  title   = {Expert-Aware Refusal Steering},
  author  = {Marbut, Anna C. and Wheeler, Travis J. and Olson, Daniel R.},
  journal = {Preprint under review},
  year    = {2026}
}
```
