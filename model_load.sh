#!/usr/bin/env bash
set -euo pipefail
source ~/.bashrc

MODEL_DIR=$HF_HOME/models   # central place for local copies
mkdir -p "$MODEL_DIR"

# Alignment_1 machine
# Base
huggingface-cli download meta-llama/Meta-Llama-3.1-8B \
  --local-dir "$MODEL_DIR/Meta-Llama-3.1-8B" \
  --include "config.json" "generation_config.json" "tokenizer.*" "*.safetensors"

# Instruct
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir "$MODEL_DIR/Meta-Llama-3.1-8B-Instruct" \
  --include "config.json" "generation_config.json" "tokenizer.*" "*.safetensors"

echo "Done. Models in $MODEL_DIR:"
du -sh "$MODEL_DIR"/*

#alignment_2 machine
#!/usr/bin/env bash
set -euo pipefail
source ~/.bashrc

MODEL_DIR=$HF_HOME/models
mkdir -p "$MODEL_DIR"


huggingface-cli download openai/gpt-oss-20b \
  --local-dir "$MODEL_DIR/oss-20b" \
  --include "config.json" "generation_config.json" "tokenizer.*" "*.safetensors"

echo "Done. Model in $MODEL_DIR/oss-20b"
du -sh "$MODEL_DIR/oss-20b"

#alignment_3 machine
#!/usr/bin/env bash
set -euo pipefail
source ~/.bashrc

MODEL_DIR=$HF_HOME/models
mkdir -p "$MODEL_DIR"

huggingface-cli download mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --local-dir "$MODEL_DIR/Mixtral-8x7B-Instruct-v0.1" \
  --include "config.json" "generation_config.json" "tokenizer.*" "*.safetensors"

echo "Done. Model in $MODEL_DIR/Mixtral-8x7B-Instruct-v0.1"
du -sh "$MODEL_DIR/Mixtral-8x7B-Instruct-v0.1"

