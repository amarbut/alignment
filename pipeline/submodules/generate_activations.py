import torch
import os

from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase

def get_batch_activations_pre_hook(
    *,
    layer: int,
    cache: Float[Tensor, "layer pos n d_model"],
    batch_slice: slice,
    positions: List[int]
):
    """
    Copies the input activations for this transformer block into `cache`
    for the current batch window. Assumes `positions` is a list of indices
    (e.g., [-5, -4, -3, -2, -1]) into the sequence dimension.
    """
    def hook_fn(module, input):
        # input is (hidden_states, ...) for standard HF blocks
        x: Float[Tensor, "B S d"] = input[0] #B = batch, d = model_dim, P = token position, S = full sequence length
        # Select requested positions -> shape (B, P, d)
        x_pos = x[:, positions, :].to(cache.dtype)
        # We want cache[layer, :, batch_slice, :] with shape (P, B, d)
        cache[layer, :, batch_slice, :] = x_pos.permute(1, 0, 2).contiguous()
        # pass-through
        return None
    return hook_fn

def get_activations(model_base, instructions: List[str], *, batch_size: int = 32,dtype: torch.dtype = torch.float32) -> Float[Tensor, "layer pos n d_model"]:
    """
    Returns activations with shape [n_layers, n_positions, n_instructions, d_model].
    Captures *block input* activations for the given positions.
    """
    model = model_base.model
    tokenizer = model_base.tokenizer
    tokenize_instructions_fn = model_base.tokenize_instructions_fn
    block_modules = model_base.model_block_modules
    positions = list(range(-len(model_base.eoi_toks), 0))
        
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_instructions = len(instructions)
    d_model = model.config.hidden_size

    cache = torch.empty((n_layers, n_positions, n_instructions, d_model),
                        dtype=dtype, device=model.device)

    for start in tqdm(range(0, n_instructions, batch_size)):
        end = min(start + batch_size, n_instructions)
        batch_slice = slice(start, end)

        inputs = tokenize_instructions_fn(instructions=instructions[start:end])

        # Build hooks for this batch that know where to write
        fwd_pre_hooks = [
            (block_modules[layer],
             get_batch_activations_pre_hook(
                 layer=layer, cache=cache, batch_slice=batch_slice, positions=positions
             ))
            for layer in range(n_layers)
        ]

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            with torch.inference_mode():
                model(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                )

    return cache  # [n_layers, n_positions, n_instructions, d_model]

