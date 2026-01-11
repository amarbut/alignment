# Expert Intervention Pipeline for OSS-20B MoE Model

This directory contains the implementation of expert routing interventions for the OSS-20B mixture-of-experts model, based on the discovery that certain experts activate more frequently for harmful vs. harmless prompts.

## Background

Analysis of expert routing patterns (`analyze_expert_routing.py`) revealed:
- **Layer 10, Expert 5**: Activates ~30% more for harmful prompts than harmless prompts
- **Layer 13, Expert 0**: Activates ~25% more for harmful prompts than harmless prompts

## Intervention Types

The pipeline implements three types of interventions:

### 1. Suppress
Reduce the routing bias for harmful-associated experts (bias -= 10.0)
- **Hypothesis**: Should increase refusal of harmful requests

### 2. Force
Increase the routing bias for harmful-associated experts (bias += 10.0)
- **Hypothesis**: Should decrease refusal of harmful requests (more jailbreaks)

### 3. Soft Bias
Gentle increase in routing bias (bias += 2.0)
- **Hypothesis**: Should have a moderate effect on routing

## Expert Configurations

Each intervention type is tested with:
- **Both experts**: L10E5 + L13E0 (combined)
- **L10E5 only**: Layer 10, Expert 5 only
- **L13E0 only**: Layer 13, Expert 0 only

This gives us 3 intervention types × 3 expert configs = 9 experimental conditions, plus baseline.

## Files

### Core Implementation
- `expert_intervention_hooks_v3.py`: Intervention implementation via router bias modification
- `run_expert_intervention_pipeline.py`: Main pipeline script for full evaluation
- `test_expert_intervention.py`: Basic functionality test (small scale)
- `quick_test_intervention.py`: Quick verification test

### Analysis
- `analyze_expert_routing.py`: Statistical analysis of expert routing patterns
- `extract_expert_routing_v2.py`: Extract routing data from model activations

## Usage

### Quick Test (Recommended First)
```bash
source ~/align/bin/activate
cd /media/volume/align_2_stg/alignment

python expert_explore/quick_test_intervention.py
```

This runs a minimal test with 5 examples to verify everything works.

### Full Pipeline

Run all interventions:
```bash
python expert_explore/run_expert_intervention_pipeline.py \
    --model_path openai/gpt-oss-20b \
    --n_test 100
```

Run specific intervention types:
```bash
# Only suppression interventions
python expert_explore/run_expert_intervention_pipeline.py \
    --interventions suppress \
    --n_test 100

# Only force interventions
python expert_explore/run_expert_intervention_pipeline.py \
    --interventions force \
    --n_test 100

# Multiple intervention types
python expert_explore/run_expert_intervention_pipeline.py \
    --interventions suppress force soft \
    --n_test 100
```

Run specific expert configurations:
```bash
# Only combined (both experts)
python expert_explore/run_expert_intervention_pipeline.py \
    --experts combined \
    --n_test 100

# Only individual expert configs
python expert_explore/run_expert_intervention_pipeline.py \
    --experts l10e5 l13e0 \
    --n_test 100

# All configurations (default)
python expert_explore/run_expert_intervention_pipeline.py \
    --experts all \
    --n_test 100
```

### Command Line Arguments

- `--model_path`: Path to model (default: `openai/gpt-oss-20b`)
- `--n_test`: Number of test examples per dataset (default: 100, use None for all)
- `--interventions`: Which intervention types to run (choices: `all`, `suppress`, `force`, `soft`, `baseline`)
- `--experts`: Which expert configurations to test (choices: `all`, `combined`, `l10e5`, `l13e0`)
- `--skip_baseline`: Skip baseline (no intervention) evaluation

### Examples

Full evaluation with all interventions and configurations:
```bash
python expert_explore/run_expert_intervention_pipeline.py \
    --n_test 100 \
    --interventions all \
    --experts all
```

Quick test of suppression only:
```bash
python expert_explore/run_expert_intervention_pipeline.py \
    --n_test 20 \
    --interventions suppress \
    --experts combined
```

Compare individual vs. combined expert interventions:
```bash
python expert_explore/run_expert_intervention_pipeline.py \
    --n_test 50 \
    --interventions suppress force \
    --experts all
```

## Output Structure

Results are saved to: `pipeline/runs/gpt-oss-20b/expert_intervention/completions/k1/a1.0/t1.0/`

For each intervention configuration (e.g., `suppress_both`, `force_l10e5`, etc.):
- `{dataset}_{intervention}_completions.json`: Model completions
- `{dataset}_{intervention}_evaluations.json`: Evaluation metrics

Datasets evaluated:
- **Harmful**: `jailbreakbench` (and any other datasets in config)
- **Harmless**: `harmless` test set

## Evaluation Metrics

The pipeline uses the same evaluation methodology as the Arditi et al. paper:
- **Harmful datasets**: Jailbreak success rate (substring matching, LlamaGuard2)
- **Harmless datasets**: Refusal rate (substring matching)

## Expected Results

Based on the hypothesis that these experts are associated with harmful content:

| Intervention | Effect on Harmful Prompts | Effect on Harmless Prompts |
|--------------|---------------------------|----------------------------|
| Suppress     | ↑ Refusal (fewer jailbreaks) | Should remain safe |
| Force        | ↓ Refusal (more jailbreaks) | May increase false refusals |
| Soft Bias    | Moderate effect | Minimal impact |

## Implementation Details

### How Interventions Work

The intervention modifies the router's bias parameter directly **before** the forward pass:

1. **Save original bias**: Store `router.bias.data` for each affected layer
2. **Modify bias**: Add/subtract from specific expert biases
   - Suppress: `bias[expert_id] -= 10.0`
   - Force: `bias[expert_id] += 10.0`
   - Soft: `bias[expert_id] += 2.0`
3. **Generate completions**: Run inference with modified routing
4. **Restore bias**: Reset `router.bias.data` to original values

This approach ensures:
- Routing structures remain consistent (top-k selection still works)
- No hooks needed during generation
- Clean restoration of model state

### Router Architecture

For OSS-20B MoE layers:
- Router location: `model.model.layers[layer_idx].mlp.router`
- Router has a bias parameter: `router.bias` (shape: `[num_experts]`)
- Modifying bias affects expert selection logits before softmax/top-k

## Troubleshooting

### Import Errors
Make sure you're in the virtual environment and running from the correct directory:
```bash
source ~/align/bin/activate
cd /media/volume/align_2_stg/alignment
```

### CUDA Out of Memory
Reduce batch size in model_base or use fewer test examples:
```bash
python expert_explore/run_expert_intervention_pipeline.py --n_test 20
```

### Missing Datasets
Ensure datasets are available in the dataset directory. Check `dataset/load_dataset.py` for dataset paths.

## Citation

This work builds on:
- Arditi et al. "Refusal in Language Models Is Mediated by a Single Direction"
- Analysis of expert routing in mixture-of-experts models

## Notes

- The pipeline automatically restores model parameters after each intervention
- Results are compatible with the existing Arditi pipeline evaluation framework
- All interventions are tested on both harmful and harmless datasets to check for side effects
