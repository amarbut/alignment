#!/bin/bash
# Controlled experiments for MoE expert dropout fine-tuning
# Run all three experimental conditions with consistent settings

set -e  # Exit on error

# Activate virtual environment
source ~/align/bin/activate

# Set HuggingFace cache directories
export HF_HOME=/media/volume/align_2_stg/hf
export TRANSFORMERS_CACHE=/media/volume/align_2_stg/hf/transformers
export HF_HUB_CACHE=/media/volume/align_2_stg/hf/hub
export HUGGINGFACE_HUB_CACHE=/media/volume/align_2_stg/hf/hub

# Common settings across all experiments
MODEL="openai/gpt-oss-20b"
EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LR=5e-5
MAX_SAMPLES=1000  # Set to smaller number for quick testing, None for full dataset
DROPOUT_RATE=0.3
SEED=42

echo "========================================================================"
echo "CONTROLLED MoE FINE-TUNING EXPERIMENTS"
echo "========================================================================"
echo "Model: $MODEL"
echo "Epochs: $EPOCHS"
echo "Max samples: $MAX_SAMPLES"
echo "Expert dropout rate: $DROPOUT_RATE"
echo ""
echo "Running three experimental conditions:"
echo "  1. ROUTER-ONLY: Train only routing parameters"
echo "  2. EXPERT-ONLY: Train only expert weights (via LoRA)"
echo "  3. COMBINED: Train both routers and experts"
echo "========================================================================"
echo ""

# Experiment 1: Router-only training
echo "========================================================================"
echo "EXPERIMENT 1: ROUTER-ONLY TRAINING"
echo "========================================================================"
python finetune_moe_controlled.py \
    --training_mode router \
    --model_name "$MODEL" \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --expert_dropout_rate $DROPOUT_RATE \
    --max_samples $MAX_SAMPLES \
    --seed $SEED \
    --output_dir ./checkpoints/exp1_router_only

echo ""
echo "Experiment 1 complete!"
echo ""

# Experiment 2: Expert-only training
echo "========================================================================"
echo "EXPERIMENT 2: EXPERT-ONLY TRAINING"
echo "========================================================================"
python finetune_moe_controlled.py \
    --training_mode expert \
    --model_name "$MODEL" \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --expert_dropout_rate $DROPOUT_RATE \
    --lora_rank 8 \
    --lora_alpha 16 \
    --max_samples $MAX_SAMPLES \
    --seed $SEED \
    --output_dir ./checkpoints/exp2_expert_only

echo ""
echo "Experiment 2 complete!"
echo ""

# Experiment 3: Combined training
echo "========================================================================"
echo "EXPERIMENT 3: COMBINED TRAINING"
echo "========================================================================"
python finetune_moe_controlled.py \
    --training_mode combined \
    --model_name "$MODEL" \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --expert_dropout_rate $DROPOUT_RATE \
    --lora_rank 8 \
    --lora_alpha 16 \
    --max_samples $MAX_SAMPLES \
    --seed $SEED \
    --output_dir ./checkpoints/exp3_combined

echo ""
echo "Experiment 3 complete!"
echo ""

echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - ./checkpoints/exp1_router_only/"
echo "  - ./checkpoints/exp2_expert_only/"
echo "  - ./checkpoints/exp3_combined/"
echo ""
echo "Next steps:"
echo "  1. Evaluate each model on harmful prompts"
echo "  2. Test robustness with expert interventions"
echo "  3. Compare routing patterns across conditions"
echo "========================================================================"
