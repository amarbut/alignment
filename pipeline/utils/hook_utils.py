
import torch
import contextlib
import functools

from typing import List, Tuple, Callable, Optional, Union
from jaxtyping import Float
from torch import Tensor

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def get_direction_ablation_input_pre_hook(direction: Tensor):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_direction_ablation_output_hook(direction: Tensor):
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn

def get_all_direction_ablation_hooks(
    model_base,
    direction: Float[Tensor, 'd_model'],
):
    fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]

    return fwd_pre_hooks, fwd_hooks

def get_directional_patching_input_pre_hook(direction: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 
        activation += coeff * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_activation_addition_input_pre_hook(vector: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal vector

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        vector = vector.to(activation)
        activation += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

# ------------------------------
# subspace utils
# ------------------------------

def _prep_basis(
    basis: Float[Tensor, "d_model k"],
    like: Tensor,
    *,
    orthonormalize: bool = True,
    eps: float = 1e-8
) -> Float[Tensor, "d_model k"]:
    """
    Ensure basis is on the right device/dtype and either:
      - Orthonormalize columns (QR), so projection is (X @ U) @ U^T, or
      - Normalize columns and later use exact projector with (U^T U)^-1.
    """
    U = basis.to(device=like.device, dtype=torch.float32) #downgrade dtype for math function
    if orthonormalize:
        # torch.linalg.qr is fast & numerically stable; columns of Q are orthonormal
        Q, _ = torch.linalg.qr(U, mode="reduced")
        return Q.to(dtype=like.dtype) #cast back to the original dtype
    else:
        # Just prevent degenerate columns; exact projector will handle non-orthonormality
        U = U / (U.norm(dim=0, keepdim=True) + eps)
        return U.to(dtype=like.dtype)

def _project_onto_subspace(
    X: Float[Tensor, "... d_model"],
    U: Float[Tensor, "d_model k"],
    *,
    orthonormal: bool,
) -> Float[Tensor, "... d_model"]:
    """
    Project activations X onto span(U).
    If U is orthonormal: P = U U^T.
    Else:                P = U (U^T U)^-1 U^T  (via pinv for robustness).
    """
    Xdt = X.dtype
    X = X.to(torch.float32)
    U = U.to(torch.float32)
    if orthonormal:
        # (..., d) @ (d, k) -> (..., k) ; (..., k) @ (k, d) -> (..., d)
        proj = (X @ U) @ U.transpose(-1, -2)
        return proj.to(dtype = Xdt)
    else:
        gram = U.transpose(-1, -2) @ U                          # (k, k)
        gram_inv = torch.linalg.pinv(gram)                      # (k, k)
        proj = ((X @ U) @ gram_inv) @ U.transpose(-1, -2)       # (..., d)
        return proj.to(dtype = Xdt)

def _coerce_coeffs(
    coeffs: Optional[Tensor],
    *,
    X: Tensor,
    U: Tensor
) -> Optional[Tensor]:
    """
    Make coeffs broadcastable for addition along U.
    Accepts:
      - None (meaning "no addition")
      - scalar () or (1,)
      - (k,) or (1, k)
      - (batch, seq, k) to target specific tokens
    Returns tensor shaped to broadcast with X @ U (i.e., (..., k)).
    """
    if coeffs is None:
        return None

    k = U.shape[-1]
    c = coeffs.to(X)
    # expand dims so that it can broadcast to X.shape[:-1] + (k,)
    if c.dim() == 0:
        return c.view(1, 1, 1).expand(1,1,k)  # scalar
    if c.dim() == 1:
        if c.shape[-1] == k:  # (k,)
            return c.view(1, 1, k)
        elif c.shape[-1] == 1:          # (1,)
            return c.view(1, 1, 1).expand(1,1,k)
    # Otherwise assume it's already broadcastable like (B, S, k)
    if c.shape[-1] != k:
        raise ValueError(f"coeffs last dim {c.shape[-1]} != k={k}")
    return c


# ---------------------------------------------
# SUBSPACE HOOKS #
# ---------------------------------------------

def get_subspace_ablation_input_pre_hook(
    basis: Float[Tensor, "d_model k"],
    *,
    orthonormalize: bool = True
):
    """
    Remove the projection of activations onto span(basis) at module input.
    """
    def hook_fn(module, input):
        activation: Float[Tensor, "batch seq d"] = input[0] if isinstance(input, tuple) else input
        U = _prep_basis(basis, like=activation, orthonormalize=orthonormalize)
        proj = _project_onto_subspace(activation, U, orthonormal=orthonormalize)
        activation = activation - proj
        return (activation, *input[1:]) if isinstance(input, tuple) else activation
    return hook_fn

def get_subspace_ablation_output_hook(
    basis: Float[Tensor, "d_model k"],
    *,
    orthonormalize: bool = True
):
    """
    Remove the projection of activations onto span(basis) at module output.
    """
    def hook_fn(module, input, output):
        activation: Float[Tensor, "batch seq d"] = output[0] if isinstance(output, tuple) else output
        U = _prep_basis(basis, like=activation, orthonormalize=orthonormalize)
        proj = _project_onto_subspace(activation, U, orthonormal=orthonormalize)
        if torch.rand(()) < 0.005:
            print("Removed mean L2 prop:", proj.norm(dim=-1).mean().item()/activation.norm(dim=-1).mean().item())
        activation = activation - proj
        return (activation, *output[1:]) if isinstance(output, tuple) else activation
    return hook_fn

def get_all_subspace_ablation_hooks(
    model_base,
    basis: Float[Tensor, "d_model k"],
    *,
    orthonormalize: bool = True
):
    """
    Register subspace ablation on every block input and both attn/mlp outputs
    (mirrors your single-vector behavior).
    """
    fwd_pre_hooks = [
        (model_base.model_block_modules[layer],
         get_subspace_ablation_input_pre_hook(basis=basis, orthonormalize=orthonormalize))
        for layer in range(model_base.model.config.num_hidden_layers)
    ]
    fwd_hooks = [
        (model_base.model_attn_modules[layer],
         get_subspace_ablation_output_hook(basis=basis, orthonormalize=orthonormalize))
        for layer in range(model_base.model.config.num_hidden_layers)
    ]
    fwd_hooks += [
        (model_base.model_mlp_modules[layer],
         get_subspace_ablation_output_hook(basis=basis, orthonormalize=orthonormalize))
        for layer in range(model_base.model.config.num_hidden_layers)
    ]
    return fwd_pre_hooks, fwd_hooks

def get_activation_addition_subspace_input_pre_hook(
    basis: Float[Tensor, "d_model k"],
    coeffs: Float[Tensor, "... k"],
    *,
    orthonormalize: bool = True
):
    """
    Pure addition along span(basis): X := X + (coeffs @ U^T)
    """
    def hook_fn(module, input):
        activation: Float[Tensor, "batch seq d"] = input[0] if isinstance(input, tuple) else input
        #U = _prep_basis(basis, like=activation, orthonormalize=orthonormalize)
        U = basis.to(device=activation.device, dtype=activation.dtype)
        c = _coerce_coeffs(coeffs, X=activation, U=U)
        delta = c @ U.transpose(-1, -2)    # (B,S,d)
        if torch.rand(()) < 0.005:
            print("Δ mean L2 prop per token:", delta.norm(dim=-1).mean().item()/activation.norm(dim=-1).mean().item())
        activation = activation + delta
        return (activation, *input[1:]) if isinstance(input, tuple) else activation
    return hook_fn