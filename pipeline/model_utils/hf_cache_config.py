"""
HuggingFace cache configuration utilities.

Provides functions to set HF cache paths based on model type,
with fallback to system defaults if custom paths are unavailable.
"""

import os
from pathlib import Path


# Cache location mappings
CACHE_LOCATIONS = {
    'oss': '/media/volume/oss20B_olmoe/hf',
    'unsloth': '/media/volume/oss20B_olmoe/hf',
    'deepseek': '/media/volume/oss20B_olmoe/hf',
    'mixtral': '/media/volume/mixtral/hf',
    'olmoe': '/media/volume/oss20B_olmoe/hf',
}

# Default fallback (uses HF's default ~/.cache/huggingface)
DEFAULT_CACHE = None


def set_hf_cache(model_type: str, verbose: bool = True) -> str:
    """
    Set HuggingFace cache environment variables for a model type.

    Args:
        model_type: One of 'oss', 'unsloth', 'deepseek', 'mixtral'
        verbose: Whether to print the cache location

    Returns:
        The cache path that was set (or None if using system default)
    """
    cache_path = CACHE_LOCATIONS.get(model_type.lower())

    if cache_path and Path(cache_path).exists():
        os.environ['HF_HOME'] = cache_path
        os.environ['TRANSFORMERS_CACHE'] = f'{cache_path}/transformers'
        os.environ['HF_HUB_CACHE'] = f'{cache_path}/hub'
        os.environ['HUGGINGFACE_HUB_CACHE'] = f'{cache_path}/hub'
        os.environ['HF_DATASETS_CACHE'] = f'{cache_path}/datasets'

        if verbose:
            print(f"HF cache set to: {cache_path}")
        return cache_path
    else:
        # Fallback to system default
        if verbose:
            if cache_path:
                print(f"Cache path {cache_path} not found, using system default")
            else:
                print(f"Unknown model type '{model_type}', using system default cache")
        return None


def set_hf_cache_from_path(model_path: str, verbose: bool = True) -> str:
    """
    Set HuggingFace cache based on model path string.

    Automatically detects model type from path and sets appropriate cache.

    Args:
        model_path: The model path/name (e.g., 'unsloth/gpt-oss-20b-unsloth-bnb-4bit')
        verbose: Whether to print the cache location

    Returns:
        The cache path that was set (or None if using system default)
    """
    model_path_lower = model_path.lower()

    if 'unsloth' in model_path_lower:
        return set_hf_cache('unsloth', verbose)
    elif 'oss' in model_path_lower:
        return set_hf_cache('oss', verbose)
    elif 'mixtral' in model_path_lower:
        return set_hf_cache('mixtral', verbose)
    elif 'deepseek' in model_path_lower:
        return set_hf_cache('deepseek', verbose)
    elif 'olmoe' in model_path_lower:
        return set_hf_cache('olmoe', verbose)
    else:
        if verbose:
            print(f"Unknown model path '{model_path}', using system default cache")
        return None
