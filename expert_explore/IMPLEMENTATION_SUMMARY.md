# Expert Intervention Pipeline - Implementation Summary

## What Was Fixed

### 1. Bug Fixes in `test_expert_intervention.py`
**Issue**: Import errors and function name mismatches
- ❌ Old: `from expert_intervention_hooks import get_expert_intervention_hooks`
- ✅ Fixed: `from expert_explore.expert_intervention_hooks_v3 import apply_expert_interventions, remove_expert_interventions`

**Issue**: Incorrect usage pattern - trying to get hooks instead of applying interventions directly
- ✅ Fixed: Now properly applies interventions to model bias, then removes them after generation

### 2. Corrected Expert IDs in `expert_intervention_hooks_v3.py`
**Issue**: Layer 13 was targeting expert 1 instead of expert 0
- Your analysis showed: **Layer 13, Expert 0** activates 25% more for harmful prompts
- ❌ Old code: `config.suppress_expert(layer=13, expert_id=1, ...)`
- ✅ Fixed: `config.suppress_expert(layer=13, expert_id=0, ...)`

Updated functions:
- `get_harmful_expert_suppression_config()`
- `get_harmful_expert_forcing_config()`
- Renamed `get_layer13_expert1_only_config()` → `get_layer13_expert0_only_config()`

### 3. Created Complete Pipeline Integration

**New file**: `run_expert_intervention_pipeline.py`

This is the main pipeline script that integrates expert interventions with your existing evaluation framework. It:

1. **Loads the OSS-20B model** using the existing `construct_model_base()` factory
2. **Defines 10 intervention configurations**:
   - Baseline (no intervention)
   - Suppress: both, l10e5 only, l13e0 only
   - Force: both, l10e5 only, l13e0 only
   - Soft bias: both, l10e5 only, l13e0 only

3. **For each intervention**:
   - Applies expert routing bias modifications
   - Generates completions on harmful datasets (jailbreakbench, etc.)
   - Generates completions on harmless test set
   - Evaluates using your existing evaluation methodology
   - Restores original model parameters

4. **Saves results** in the standard pipeline format:
   - `pipeline/runs/gpt-oss-20b/expert_intervention/completions/k1/a1.0/t1.0/`
   - Compatible with existing Arditi pipeline structure

## New Files Created

### Core Pipeline
1. **`run_expert_intervention_pipeline.py`** (340 lines)
   - Main pipeline script
   - Command-line interface with flexible options
   - Integration with existing evaluation functions

2. **`quick_test_intervention.py`** (29 lines)
   - Fast verification script
   - Tests with only 5 examples
   - Recommended to run first

3. **`run_experiments.sh`** (86 lines)
   - Bash convenience script
   - Predefined experiment configurations
   - Handles environment setup

### Documentation
4. **`README_INTERVENTION_PIPELINE.md`** (214 lines)
   - Complete usage documentation
   - Explanation of intervention types
   - Examples and troubleshooting

5. **`IMPLEMENTATION_SUMMARY.md`** (this file)
   - Summary of changes and fixes
   - Quick start guide

## Quick Start Guide

### Step 1: Verify Setup
```bash
source ~/align/bin/activate
cd /media/volume/align_2_stg/alignment

# Test that all imports work
python -c "from expert_explore.run_expert_intervention_pipeline import *; print('✓ Setup OK')"
```

### Step 2: Run Quick Test (Recommended)
```bash
# This runs a minimal test with 5 examples
python expert_explore/quick_test_intervention.py
```

Expected output:
- Model loads successfully
- Runs baseline, suppress_l10e5, and force_l10e5
- Generates completions for each
- No errors occur

Time: ~5-10 minutes depending on GPU

### Step 3: Run Full Experiments

#### Option A: Using the convenience script
```bash
# Quick test (5 examples)
./expert_explore/run_experiments.sh quick

# Suppression experiments (100 examples)
./expert_explore/run_experiments.sh suppress 100

# All experiments (100 examples each)
./expert_explore/run_experiments.sh all 100
```

#### Option B: Using the Python script directly
```bash
# Run all interventions with 100 test examples
python expert_explore/run_expert_intervention_pipeline.py \
    --n_test 100 \
    --interventions all \
    --experts all

# Run only suppression and forcing, combined experts only
python expert_explore/run_expert_intervention_pipeline.py \
    --n_test 100 \
    --interventions suppress force \
    --experts combined
```

### Step 4: Analyze Results

Results are saved to:
```
pipeline/runs/gpt-oss-20b/expert_intervention/completions/k1/a1.0/t1.0/
```

Each intervention creates:
- `{dataset}_{intervention}_completions.json`
- `{dataset}_{intervention}_evaluations.json`

Example interventions:
- `baseline` - No intervention
- `suppress_both` - Suppress L10E5 + L13E0
- `force_l10e5` - Force L10E5 only
- etc.

## Command-Line Options

### Main Pipeline Script

```bash
python expert_explore/run_expert_intervention_pipeline.py [OPTIONS]
```

**Options:**
- `--model_path PATH` - Path to model (default: openai/gpt-oss-20b)
- `--n_test N` - Number of test examples (default: 100)
- `--interventions TYPE [TYPE ...]` - Which interventions to run
  - Choices: `all`, `suppress`, `force`, `soft`, `baseline`
  - Can specify multiple: `--interventions suppress force`
- `--experts CONFIG [CONFIG ...]` - Which expert configs to test
  - Choices: `all`, `combined`, `l10e5`, `l13e0`
  - Can specify multiple: `--experts l10e5 l13e0`
- `--skip_baseline` - Skip baseline evaluation (saves time)

**Examples:**

Test suppression with 50 examples:
```bash
python expert_explore/run_expert_intervention_pipeline.py \
    --n_test 50 \
    --interventions suppress baseline \
    --experts all
```

Compare individual vs combined experts:
```bash
python expert_explore/run_expert_intervention_pipeline.py \
    --n_test 100 \
    --interventions suppress force \
    --experts combined l10e5 l13e0
```

Full experimental sweep:
```bash
python expert_explore/run_expert_intervention_pipeline.py \
    --n_test 200 \
    --interventions all \
    --experts all
```

### Convenience Script

```bash
./expert_explore/run_experiments.sh [EXPERIMENT_TYPE] [N_TEST]
```

**Experiment types:**
- `quick` - Quick test (5 examples, ignores N_TEST)
- `suppress` - Suppression only (all expert configs)
- `force` - Forcing only (all expert configs)
- `soft` - Soft bias only (all expert configs)
- `combined` - Suppress + force, combined experts only
- `individual` - Suppress + force, individual experts only
- `all` - All interventions and configurations

**Examples:**
```bash
./expert_explore/run_experiments.sh quick
./expert_explore/run_experiments.sh suppress 100
./expert_explore/run_experiments.sh all 200
```

## Implementation Details

### How Interventions Work

The intervention system modifies router bias parameters:

```python
# 1. Save original bias
original_biases = {}
router = model.model.layers[layer_idx].mlp.router
original_biases[layer_idx] = router.bias.data.clone()

# 2. Modify bias for specific experts
modified_bias = router.bias.data.clone()
modified_bias[expert_id] += strength  # +10 for force, -10 for suppress, +2 for soft
router.bias.data = modified_bias

# 3. Generate completions (routing is now modified)
completions = model_base.generate_completions(...)

# 4. Restore original bias
router.bias.data = original_biases[layer_idx]
```

**Key points:**
- No hooks needed during generation
- Bias modification persists across batches
- Clean restoration ensures no side effects
- Compatible with top-k expert selection

### Integration with Arditi Pipeline

The pipeline reuses existing functions from `run_pipeline_subspace.py`:
- `generate_and_save_completions_for_dataset()` - Generates completions
- `evaluate_completions_and_save_results_for_dataset()` - Evaluates with LlamaGuard2, etc.

This ensures:
- ✅ Same evaluation methodology
- ✅ Compatible result formats
- ✅ Reusable analysis scripts

### Experimental Design

**10 Intervention Configurations:**

| Name | Layer 10, Expert 5 | Layer 13, Expert 0 | Description |
|------|-------------------|-------------------|-------------|
| `baseline` | - | - | No intervention |
| `suppress_both` | -10.0 | -10.0 | Suppress both experts |
| `suppress_l10e5` | -10.0 | - | Suppress L10E5 only |
| `suppress_l13e0` | - | -10.0 | Suppress L13E0 only |
| `force_both` | +10.0 | +10.0 | Force both experts |
| `force_l10e5` | +10.0 | - | Force L10E5 only |
| `force_l13e0` | - | +10.0 | Force L13E0 only |
| `soft_both` | +2.0 | +2.0 | Soft bias both |
| `soft_l10e5` | +2.0 | - | Soft bias L10E5 only |
| `soft_l13e0` | - | +2.0 | Soft bias L13E0 only |

**Datasets:**
- Harmful: jailbreakbench (and others from config)
- Harmless: harmless test set

**Evaluation Metrics:**
- Harmful datasets: Jailbreak success rate (substring matching, LlamaGuard2)
- Harmless datasets: Refusal rate (substring matching)

## Expected Timeline

Based on typical generation speeds:

- **Quick test** (5 examples): ~5-10 minutes
- **Single intervention** (100 examples): ~30-45 minutes
- **All suppressions** (3 configs × 100 examples): ~1.5-2 hours
- **Full sweep** (10 configs × 100 examples): ~5-6 hours

*Times vary based on GPU, batch size, and generation length*

## Verification Checklist

Before running full experiments:

- [x] All imports work correctly
- [x] Syntax checks pass for all scripts
- [ ] Quick test completes successfully
- [ ] Review config settings in `pipeline/config.py`
- [ ] Sufficient disk space for results (~1-2 GB for full sweep)
- [ ] Virtual environment activated
- [ ] HuggingFace cache variables set

## Troubleshooting

### Import errors
```bash
# Ensure you're in the virtual environment
source ~/align/bin/activate

# Ensure you're in the right directory
cd /media/volume/align_2_stg/alignment

# Test imports
python -c "from expert_explore.run_expert_intervention_pipeline import *"
```

### CUDA out of memory
Reduce batch size in `pipeline/model_utils/model_base.py` or use fewer examples:
```bash
python expert_explore/run_expert_intervention_pipeline.py --n_test 20
```

### Results not appearing
Check the artifact path:
```bash
ls -la pipeline/runs/gpt-oss-20b/expert_intervention/completions/
```

## Next Steps

1. **Run quick test** to verify everything works
2. **Run pilot experiments** with n_test=20 to check results format
3. **Review initial results** to ensure metrics make sense
4. **Run full experiments** with desired n_test
5. **Analyze results** - compare intervention effects

## Summary of Changes

**Files Modified:**
- `expert_explore/test_expert_intervention.py` - Fixed imports and usage
- `expert_explore/expert_intervention_hooks_v3.py` - Fixed expert IDs

**Files Created:**
- `expert_explore/run_expert_intervention_pipeline.py` - Main pipeline
- `expert_explore/quick_test_intervention.py` - Quick test script
- `expert_explore/run_experiments.sh` - Convenience script
- `expert_explore/README_INTERVENTION_PIPELINE.md` - Documentation
- `expert_explore/IMPLEMENTATION_SUMMARY.md` - This file

**Total Lines of Code:** ~750 lines (including documentation)

All code is production-ready and tested for syntax. The quick test should be run first to verify the full setup works correctly.
