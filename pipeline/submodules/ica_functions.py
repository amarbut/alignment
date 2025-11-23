import torch
import numpy as np
from sklearn.decomposition import FastICA
from typing import Dict
from tqdm import tqdm


@torch.no_grad()
def ica_whitened(X_target, X_bg, n_components=None, keep_var=0.999, eig_floor_frac=1e-3, renorm_cols=True, max_iter=1000, random_state=42):
    """
    Contrastive ICA using background whitening.

    Parallel to cpca_whitened:
    1. Whiten with respect to background (harmless) distribution
    2. Run ICA on whitened target (harmful) activations
    3. Back-map ICA components to original space
    4. Optionally renormalize for interpretability

    Args:
        X_target: Target activations (harmful) [N_h, H]
        X_bg: Background activations (harmless) [N_b, H]
        n_components: Number of ICA components (None = use all after whitening)
        keep_var: Fraction of background variance to keep during whitening
        eig_floor_frac: Eigenvalue floor fraction to prevent whitening explosion
        renorm_cols: Whether to renormalize components to unit background variance
        max_iter: Maximum iterations for ICA
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - mu_b: Background mean
            - V_b, var_b: Background PCA basis and variances (for whitening)
            - components: ICA components in original space
            - components_norm: Normalized ICA components
            - mixing: ICA mixing matrix
            - discrimination_scores: Fisher score for each component
    """
    X_target = X_target.to(torch.float64).cpu()
    X_bg = X_bg.to(torch.float64).cpu()

    eps = 1e-6

    # 1) Background PCA (for whitening)
    Xb_c = X_bg - X_bg.mean(0, keepdim=True)
    U, S, Vt = torch.linalg.svd(Xb_c, full_matrices=False)
    var_b_all = (S**2) / max(1, X_bg.shape[0]-1)
    Vb_all = Vt.T  # [H, r]

    # 2) Truncate to keep 'keep_var' fraction of variance
    order = torch.argsort(var_b_all, descending=True)
    var_b_sorted = var_b_all[order]
    Vb_sorted = Vb_all[:, order]
    cum = torch.cumsum(var_b_sorted, dim=0) / var_b_sorted.sum()
    r_keep = int(torch.searchsorted(cum, torch.tensor(keep_var), right=True).item()) + 1
    var_b = var_b_sorted[:r_keep].contiguous()
    Vb = Vb_sorted[:, :r_keep].contiguous()

    # 3) Floor small eigenvalues before inverting
    tau = eig_floor_frac * var_b.mean()
    inv_sqrt = (var_b.clamp_min(tau) + eps).rsqrt()

    # 4) Whitening
    mu_b = X_bg.mean(0, keepdim=True)

    # Truncate whitened dimensions to avoid sklearn FastICA bug
    # FastICA fails when n_features > n_samples due to internal matrix operations
    n_target_samples = X_target.shape[0]
    if r_keep > n_target_samples:
#        print(f"ICA: Truncating whitened features from {r_keep} to {n_target_samples} to match sample count")
        Vb = Vb[:, :n_target_samples]
        inv_sqrt = inv_sqrt[:n_target_samples]
        var_b = var_b[:n_target_samples]
        r_keep = n_target_samples

    def whiten(X):
        Xc = X - mu_b
        return (Xc @ Vb) * inv_sqrt  # [N, r_keep]

    Zt = whiten(X_target)  # Whitened target data [N_h, r_keep]

    # 5) ICA on whitened target
    # Convert to numpy for sklearn
    Zt_np = Zt.numpy()

    # DEBUG: Print shapes
#    print(f"DEBUG ICA: X_target.shape = {X_target.shape}, X_bg.shape = {X_bg.shape}")
#    print(f"DEBUG ICA: Vb.shape = {Vb.shape}")
#    print(f"DEBUG ICA: Zt_np.shape (whitened harmful) = {Zt_np.shape}")
#    print(f"DEBUG ICA: r_keep = {r_keep}")
#    print(f"DEBUG ICA: Zt_np.mean = {Zt_np.mean(axis=0, keepdims=True)}")

    # Center the whitened data (ICA expects zero mean when whiten=False)
    Zt_np = Zt_np - Zt_np.mean(axis=0, keepdims=True)

    # Determine valid n_components
    n_samples, n_features = Zt_np.shape
    n_components = n_features

#    if n_components is None:
#        n_components = max_components
#    else:
#        n_components = min(n_components, max_components)

    # Need at least 1 component
#    n_components = max(1, n_components)

#    print(f"DEBUG ICA: n_samples={n_samples}, n_features={n_features}, n_components={n_components}")

    ica = FastICA(
        n_components=n_components,
        max_iter=max_iter,
        random_state=random_state,
        whiten=False  # Already whitened
    )

    # S = sources [N_h, n_components], A = mixing matrix [r_keep, n_components]
    S = ica.fit_transform(Zt_np)
    A = ica.mixing_  # [r_keep, n_components]

    # Convert back to torch
    S = torch.from_numpy(S).to(torch.float64)
    A = torch.from_numpy(A).to(torch.float64)

    # 6) Back-map to original space
    # Components in whitened space are columns of A
    # Back-map: (Vb * inv_sqrt) @ A
    C = (Vb * inv_sqrt.unsqueeze(0)) @ A  # [H, n_components]

    # 7) Renormalize columns (optional)
    if renorm_cols:
        Xb0 = X_bg - mu_b
        s2 = (Xb0 @ C).var(dim=0, unbiased=True)  # [n_components]
        C_norm = C / torch.sqrt(s2.clamp_min(1e-12))
    else:
        C_norm = C

    # 8) Compute discrimination scores (Fisher criterion) for each component
    discrimination_scores = torch.zeros(n_components, dtype=torch.float64)

    for i in range(n_components):
        c = C_norm[:, i]

        # Project activations onto component
        proj_harmful = ((X_target - mu_b) @ c).numpy()
        proj_harmless = ((X_bg - mu_b) @ c).numpy()

        # Fisher score: between-var / within-var
        mu_h = proj_harmful.mean()
        mu_b_proj = proj_harmless.mean()
        var_h = proj_harmful.var(ddof=1)
        var_b_proj = proj_harmless.var(ddof=1)

        between_var = (mu_h - mu_b_proj) ** 2
        within_var = var_h + var_b_proj

        if within_var > 1e-10:
            discrimination_scores[i] = between_var / within_var
        else:
            discrimination_scores[i] = 0.0

    return {
        "mu_b": mu_b.squeeze(0),
        "V_b": Vb,
        "var_b": var_b,
        "components": C,
        "components_norm": C_norm,
        "mixing": A,
        "discrimination_scores": discrimination_scores
    }


@torch.no_grad()
def get_lp_slices(X_h_lpnh, X_s_lpnh, layer, pos):
    """Extract activations at specific layer and position."""
    Xt = X_h_lpnh[layer, pos].to(torch.float64, non_blocking=True).cpu()
    Xb = X_s_lpnh[layer, pos].to(torch.float64, non_blocking=True).cpu()
    return Xt, Xb


def fisher_score_from_ica(res):
    """
    Compute overall score for ICA result.
    Uses the top component's discrimination score (analogous to Rayleigh ratio for cPCA).
    """
    return float(res["discrimination_scores"][0].item())


@torch.no_grad()
def align_sign_by_means(C, Xt, Xb, mu_b):
    """
    Align component signs so harmful mean > harmless mean.

    Args:
        C: (H, k) ICA components
        Xt, Xb: (N, H) activations
        mu_b: (H,) background mean

    Returns:
        C_signed: Components with aligned signs
        s: Sign vector (k,) in {+1, -1}
    """
    Ht = (Xt - mu_b) @ C
    Hb = (Xb - mu_b) @ C

    diff = Ht.mean(dim=0) - Hb.mean(dim=0)
    s = torch.where(diff >= 0, 1.0, -1.0)
    C_signed = C * s

    return C_signed, s


def choose_best_ica(X_h, X_s, topk: int = 3, keep_var=0.999, eig_floor_frac=1e-3, align=False, n_components=None):
    """
    Find best ICA across all layers and positions using statistical selection.

    Parallel to choose_best_cpca, but uses ICA and ranks by Fisher discrimination score.

    Args:
        X_h: Harmful activations [L, P, N_h, H]
        X_s: Harmless activations [L, P, N_s, H]
        topk: Number of top components to keep
        keep_var: Variance to keep in whitening
        eig_floor_frac: Eigenvalue floor fraction
        align: Whether to align component signs
        n_components: Number of ICA components to extract (None = auto)

    Returns:
        layer, position, result_dict
    """
    L, P = X_h.shape[:2]
    best = (-1.0, None, None, None)  # (score, li, pi, result)

    # Auto-determine n_components if not specified
    if n_components is None:
        # Use a reasonable default based on data dimensions
        n_components = min(topk * 3, 50)  # Extract more than topk to have selection

    for li in tqdm(range(L), desc="ICA across layers"):
        for pi in range(P):
            Xt, Xb = get_lp_slices(X_h, X_s, li, pi)

            # Compute ICA
            res = ica_whitened(
                Xt, Xb,
                n_components=n_components,
                keep_var=keep_var,
                eig_floor_frac=eig_floor_frac
            )

            # Align signs if requested
            if align:
                res["components"], _ = align_sign_by_means(res["components"], Xt, Xb, res["mu_b"])
                res["components_norm"], _ = align_sign_by_means(res["components_norm"], Xt, Xb, res["mu_b"])

            # Score = Fisher score of top component
            score = fisher_score_from_ica(res)

            if score > best[0]:
                best = (score, li, pi, res)

    _, li, pi, res = best

    return li, pi, res
