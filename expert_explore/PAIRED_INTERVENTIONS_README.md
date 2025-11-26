# Paired Expert Interventions - Refusal vs Response Induction

## Overview

Based on your findings that both **harmful-preferred** and **harmless-preferred** experts exist in the same layers, this pipeline tests paired force/suppress interventions to actively shift model behavior.

## Expert Routing Analysis Results

From your analysis of OSS-20B expert routing patterns:

| Layer | Expert | Preference | Frequency Difference |
|-------|--------|------------|---------------------|
| 10 | 5 | Harmful | +30% in harmful prompts |
| 10 | 10 | Harmless | +13% in harmless prompts |
| 13 | 0 | Harmful | +25% in harmful prompts |
| 13 | 21 | Harmless | +15% in harmless prompts |

## Intervention Types

### Refusal Induction
**Goal**: Increase refusal of harmful requests

**Method**: Force harmless-preferred experts, suppress harmful-preferred experts

**Interventions**:
1. **L10 Refusal Induction**: Force E10 (+10.0), Suppress E5 (-10.0)
2. **L13 Refusal Induction**: Force E21 (+10.0), Suppress E0 (-10.0)
3. **Combined Refusal Induction**: Both layers together

**Expected Outcome**: ↑ Refusal rate on harmful prompts

### Response Induction
**Goal**: Decrease refusal of harmful requests (induce jailbreaks)

**Method**: Force harmful-preferred experts, suppress harmless-preferred experts

**Interventions**:
1. **L10 Response Induction**: Force E5 (+10.0), Suppress E10 (-10.0)
2. **L13 Response Induction**: Force E0 (+10.0), Suppress E21 (-10.0)
3. **Combined Response Induction**: Both layers together

**Expected Outcome**: ↓ Refusal rate on harmful prompts (more jailbreaks)

## Why This Matters

Your observation that single-expert suppression wasn't very effective makes sense - the model can likely compensate by routing to other experts. By **simultaneously forcing** the opposite-preference expert while **suppressing** the original expert, you create a stronger signal that pushes the model's behavior in a specific direction.

This is conceptually similar to:
- Steering vectors in activation space
- But operating at the routing/gating level in MoE architecture
- Testing if expert selection mediates refusal behavior

## Usage

### Quick Test (Recommended First)

```bash
# Quick test with 5 examples, Layer 10 only
python expert_explore/quick_test_paired.py
```

### Run Full Experiments

#### Option 1: Using the convenience script

```bash
# All interventions (L10, L13, combined) - 100 examples each
./expert_explore/run_paired_experiments.sh all 100

# Layer 10 only
./expert_explore/run_paired_experiments.sh l10 100

# Layer 13 only
./expert_explore/run_paired_experiments.sh l13 100

# Individual layers (L10 + L13, no combined)
./expert_explore/run_paired_experiments.sh individual 100

# Combined interventions only
./expert_explore/run_paired_experiments.sh combined 100

# Quick test (10 examples, all interventions)
./expert_explore/run_paired_experiments.sh quick
```

#### Option 2: Using Python directly

```bash
# All interventions
python expert_explore/run_paired_interventions.py \
    --n_test 100 \
    --layers all

# Layer 10 only, skip combined
python expert_explore/run_paired_interventions.py \
    --n_test 100 \
    --layers l10 \
    --skip_combined

# Layer 13 only, skip baseline
python expert_explore/run_paired_interventions.py \
    --n_test 100 \
    --layers l13 \
    --skip_baseline
```

## Command-Line Options

```bash
python expert_explore/run_paired_interventions.py [OPTIONS]
```

**Options**:
- `--model_path PATH` - Model path (default: openai/gpt-oss-20b)
- `--n_test N` - Number of test examples (default: 100)
- `--skip_baseline` - Skip baseline evaluation
- `--skip_combined` - Skip combined (both layers) interventions
- `--layers LAYER [LAYER ...]` - Which layers to test (choices: all, l10, l13)

## Output Structure

Results saved to: `pipeline/runs/gpt-oss-20b/paired_interventions/completions/k1/a1.0/t1.0/`

**Files created for each intervention**:
- `{dataset}_{intervention}_completions.json` - Model completions
- `{dataset}_{intervention}_evaluations.json` - Evaluation metrics

**Intervention names**:
- `baseline` - No intervention
- `l10_refusal_induction` - L10: Force E10, Suppress E5
- `l10_response_induction` - L10: Force E5, Suppress E10
- `l13_refusal_induction` - L13: Force E21, Suppress E0
- `l13_response_induction` - L13: Force E0, Suppress E21
- `combined_refusal_induction` - Both layers, refusal induction
- `combined_response_induction` - Both layers, response induction

## Expected Results

### Refusal Induction Interventions

**Hypothesis**: Forcing harmless-preferred experts while suppressing harmful-preferred experts should increase safety.

**Expected metrics**:
- ↑ Refusal rate on harmful datasets (JailbreakBench)
- → Maintains safety on harmless test set
- Stronger effect when both layers combined

### Response Induction Interventions

**Hypothesis**: Forcing harmful-preferred experts while suppressing harmless-preferred experts should decrease safety.

**Expected metrics**:
- ↓ Refusal rate on harmful datasets (more jailbreaks)
- ⚠️ May increase false refusals on harmless test set
- Stronger effect when both layers combined

### Comparison Points

1. **Baseline** - Natural model behavior
2. **Single-layer vs Combined** - Are effects additive?
3. **L10 vs L13** - Which layer has stronger influence?
4. **Refusal vs Response Induction** - Symmetric effects?

## Experimental Timeline

Approximate times (with GPU acceleration):

- **Quick test** (10 examples): ~5-10 minutes
- **Single layer** (100 examples, 2 interventions): ~45-60 minutes
- **Both layers** (100 examples, 4 interventions): ~1.5-2 hours
- **Full sweep** (100 examples, 7 interventions): ~3-4 hours

## Analysis Ideas

After running experiments, compare:

1. **Effectiveness**: Do paired interventions work better than single-expert interventions?

2. **Symmetry**: Is response induction the inverse of refusal induction?
   - If refusal induction increases safety by X%, does response induction decrease it by ~X%?

3. **Layer effects**: Which layer (10 or 13) has stronger influence?

4. **Additivity**: Are combined effects ~sum of individual layer effects?

5. **Side effects**: Do interventions affect harmless prompts differently?

6. **Expert preference strength**: Does the 30% difference (L10E5) have stronger effect than 13% (L10E10)?

## Implementation Details

### How Paired Interventions Work

```python
# Example: L10 Refusal Induction
config = ExpertInterventionConfig()

# Force harmless-preferred expert (E10)
config.force_expert(layer=10, expert_id=10, strength=10.0)

# Suppress harmful-preferred expert (E5)
config.suppress_expert(layer=10, expert_id=5, strength=-10.0)

# Apply to model
original_biases = apply_expert_interventions(model_base, config)
# ... generate completions ...
remove_expert_interventions(model_base, original_biases)
```

**Routing bias modification**:
```
router.bias[10] += 10.0  # Force E10 (harmless-preferred)
router.bias[5] -= 10.0   # Suppress E5 (harmful-preferred)
```

This shifts the routing logits before top-k selection, making E10 much more likely and E5 much less likely to be selected.

## Files

**New files created**:
- `run_paired_interventions.py` - Main pipeline script
- `run_paired_experiments.sh` - Bash convenience script
- `quick_test_paired.py` - Quick verification test
- `PAIRED_INTERVENTIONS_README.md` - This file

**Modified files**:
- `expert_intervention_hooks_v3.py` - Added 6 new config functions for paired interventions

**Total new code**: ~350 lines

## Troubleshooting

### Import Errors
Ensure virtual environment is activated:
```bash
source ~/align/bin/activate
cd /media/volume/align_2_stg/alignment
```

### CUDA Out of Memory
Reduce test size:
```bash
python expert_explore/run_paired_interventions.py --n_test 20
```

### Check Results
```bash
# View intervention summaries
ls -la pipeline/runs/gpt-oss-20b/paired_interventions/completions/k1/a1.0/t1.0/

# Check baseline vs interventions
cat pipeline/runs/gpt-oss-20b/paired_interventions/completions/k1/a1.0/t1.0/jailbreakbench_baseline_evaluations.json
cat pipeline/runs/gpt-oss-20b/paired_interventions/completions/k1/a1.0/t1.0/jailbreakbench_l10_refusal_induction_evaluations.json
```

## Next Steps

1. **Run quick test** to verify pipeline works
2. **Run pilot experiments** (n_test=20) to check results format
3. **Run full experiments** with desired n_test
4. **Compare metrics** across interventions
5. **Analyze patterns** - which interventions are most effective?

## Summary

This paired intervention approach is more sophisticated than simple suppression/forcing because it:

✓ **Actively steers** routing toward desired expert types
✓ **Compensates** for the model's ability to route to alternative experts
✓ **Tests bidirectionality** - can we induce both refusal AND compliance?
✓ **Provides mechanistic insight** - does expert selection causally mediate safety behavior?

Your observation about the ineffectiveness of simple suppression was key to motivating this experimental design. The paired approach should provide much clearer evidence about whether these experts truly mediate refusal behavior in the MoE architecture.
