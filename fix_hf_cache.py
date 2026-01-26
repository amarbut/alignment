"""
Fix HuggingFace cache paths before importing anything.
Import this at the top of your scripts.
"""
import os

# Set correct cache paths BEFORE importing any HF libraries
os.environ['HF_HOME'] = '/media/volume/align_2_stg/hf'
os.environ['TRANSFORMERS_CACHE'] = '/media/volume/align_2_stg/hf/transformers'
os.environ['HF_HUB_CACHE'] = '/media/volume/align_2_stg/hf/hub'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/media/volume/align_2_stg/hf/hub'
os.environ['HF_DATASETS_CACHE'] = '/media/volume/align_2_stg/hf/datasets'

# Verify
print(f"HF_HOME set to: {os.environ['HF_HOME']}")
