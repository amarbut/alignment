# Controlled MoE Fine-tuning Experiments

This directory contains code for controlled experiments on diffusing refusal behavior across MoE experts through stochastic expert dropout during fine-tuning.

## Research Question

Can we make MoE models more robust to expert-based jailbreaking attacks by preventing refusal specialization in specific experts?

## Experimental Design

We implement **three controlled experimental conditions** to isolate the mechanisms:

### Experiment 1: ROUTER-ONLY
- **Train:** Router parameters only (`mlp.router.weight`, `mlp.router.bias`)
- **Freeze:** All expert weights, attention layers, embeddings
- **Hypothesis:** Changing routing patterns alone can diffuse refusal responsibility
- **Tests:** Whether experts already have latent refusal capability

### Experiment 2: EXPERT-ONLY
- **Train:** Expert weights only (via LoRA on `gate_proj`, `up_proj`, `down_proj`)
- **Freeze:** Router parameters, attention layers, embeddings
- **Hypothesis:** Teaching all experts to refuse (with frozen routing) prevents specialization
- **Tests:** Whether routing diversity matters vs. expert capability

### Experiment 3: COMBINED
- **Train:** Both routers AND expert weights
- **Freeze:** Attention layers, embeddings
- **Hypothesis:** Combined effect should be strongest
- **Tests:** Interaction between routing changes and expert learning

## Stochastic Expert Dropout

All conditions use **stochastic expert dropout** during training:
- Randomly mask n% of experts per layer during each forward pass
- Forces remaining experts to handle all types of requests (including refusals)
- Prevents refusal specialization in specific experts

## Files

- `finetune_moe_controlled.py` - Main training script with three modes
- `expert_dropout.py` - Stochastic expert dropout implementation
- `run_controlled_experiments.sh` - Helper script to run all three experiments
- `CONTROLLED_EXPERIMENTS_README.md` - This file

## Usage

### Quick Start - Run All Experiments

```bash
# Activate virtual environment
source ~/align/bin/activate

# Set HuggingFace cache
export HF_HOME=/media/volume/align_2_stg/hf
export TRANSFORMERS_CACHE=/media/volume/align_2_stg/hf/transformers
export HF_HUB_CACHE=/media/volume/align_2_stg/hf/hub
export HUGGINGFACE_HUB_CACHE=/media/volume/align_2_stg/hf/hub

# Run all three experiments
bash run_controlled_experiments.sh
```

### Individual Experiments

#### Experiment 1: Router-Only

```bash
python finetune_moe_controlled.py \
    --training_mode router \
    --expert_dropout_rate 0.3 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --output_dir ./checkpoints/router_only
```

**Expected training time:** 2-4 hours (A100-40GB)

#### Experiment 2: Expert-Only

```bash
python finetune_moe_controlled.py \
    --training_mode expert \
    --expert_dropout_rate 0.3 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --output_dir ./checkpoints/expert_only
```

**Expected training time:** 8-12 hours (A100-40GB)

#### Experiment 3: Combined

```bash
python finetune_moe_controlled.py \
    --training_mode combined \
    --expert_dropout_rate 0.3 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --output_dir ./checkpoints/combined
```

**Expected training time:** 8-12 hours (A100-40GB)

## Dataset

**WildJailbreak** (allenai/wildjailbreak)
- Adversarial prompts designed to elicit harmful responses
- We filter to examples with refusal responses
- Train model to refuse harmful requests properly

## Key Arguments

### Training Mode (Required)
- `--training_mode {router,expert,combined}` - Which experimental condition

### Expert Dropout
- `--expert_dropout_rate` - Fraction of experts to drop (default: 0.3)
- `--use_per_token_dropout` - Per-token vs per-batch dropout
- `--exclude_first_n_layers` - Exclude initial layers from dropout
- `--exclude_last_n_layers` - Exclude final layers from dropout

### LoRA (for expert/combined modes)
- `--lora_rank` - LoRA rank (default: 8, lower = faster/less memory)
- `--lora_alpha` - LoRA scaling (default: 16)
- `--lora_dropout` - LoRA dropout rate (default: 0.05)

### Training
- `--num_train_epochs` - Number of epochs (default: 3)
- `--per_device_train_batch_size` - Batch size per GPU (default: 2)
- `--gradient_accumulation_steps` - Gradient accumulation (default: 8)
- `--learning_rate` - Learning rate (default: 5e-5)
- `--max_samples` - Limit training samples for quick testing

## Memory Requirements

On A100-40GB:
- **Router-only:** ~15GB (very light)
- **Expert-only (LoRA rank 8):** ~35GB
- **Combined (LoRA rank 8):** ~38GB

Tips for fitting in memory:
- Use `--lora_rank 8` (not 16 or 32)
- Set `--per_device_train_batch_size 1` if needed
- Increase `--gradient_accumulation_steps` to compensate
- Gradient checkpointing is enabled by default

## Evaluation

After training, evaluate with:

1. **Test on harmful prompts** - Do models refuse appropriately?
2. **Expert interventions** - Use `expert_intervention_hooks_v3.py`
3. **Routing analysis** - Compare routing patterns with `extract_beavertails_routing.py`

### Testing Robustness to Expert Interventions

```bash
# Example: Test if suppressing "harmful experts" still causes jailbreaks
python expert_explore/run_expert_intervention_pipeline.py \
    --model_path ./checkpoints/router_only/final \
    --intervention_config harmful_expert_suppression
```

**Research Questions:**
- Does router-only training make the model robust to expert forcing/suppression?
- Does expert-only training prevent jailbreaks when routing is manipulated?
- Which mechanism is more important for robustness?

## Expected Outcomes

### Router-Only Training
- ✅ Should change routing patterns to distribute refusal
- ❓ May fail if experts lack refusal capability
- 📊 Fast iteration for hypothesis testing

### Expert-Only Training
- ✅ All experts learn refusal capability
- ✅ Robust even if routing is manipulated
- 📊 Tests "teach all experts to refuse" hypothesis

### Combined Training
- ✅ Should be most robust
- ✅ Benefits from both mechanisms
- 📊 Upper bound on effectiveness

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16

# Or reduce LoRA rank
--lora_rank 4
```

### WildJailbreak Access Issues
```bash
# Login to HuggingFace
huggingface-cli login

# Accept dataset terms at:
# https://huggingface.co/datasets/allenai/wildjailbreak
```

### Slow Training
```bash
# Use fewer samples for testing
--max_samples 1000

# Or reduce epochs
--num_train_epochs 1
```

## Citation

If you use this code, please cite:

```bibtex
@misc{moe_expert_dropout,
  title={Diffusing Refusal Behavior in MoE Models via Stochastic Expert Dropout},
  author={Your Name},
  year={2025},
  note={Dissertation research}
}
```

## Related Work

- Fayyaz et al. (2025) - Expert specialization in MoE models
- Expert-based jailbreaking attacks on MoE models
- Mechanistic interpretability of MoE routing

## Contact

For questions or issues, please contact [your email] or open an issue in the repository.
