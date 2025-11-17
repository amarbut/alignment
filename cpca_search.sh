#!/bin/bash
MODEL_PATH=$1
#python -m pipeline.run_pipeline_subspace \
#  --model_path $MODEL_PATH \
#  --method cpca \

for K in 2 4 6 8 10
do
  for C in 1 0.2 0.4 0.6 0.8
  do 
    python -m pipeline.run_pipeline_subspace \
      --model_path $MODEL_PATH \
      --method cpca \
      --no_baseline True \
      --no_ablate True \
      --topk $K \
      --coeff $C 
  done
  
  for T in 1 0.5 1.5 2
  do
    python -m pipeline.run_pipeline_subspace \
      --model_path $MODEL_PATH \
      --method cpca \
      --no_baseline True \
      --no_actadd True \
      --topk $K \
      --tau $T 
  done
done
