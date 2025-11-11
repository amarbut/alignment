import torch
import pickle
import os
from pathlib import Path
from typing import Dict, List, Tuple
import torch.nn.functional as F

@torch.no_grad()
def cpca_whitened(X_target, X_bg, eps=1e-6, keep_var=0.999, eig_floor_frac=1e-3, renorm_cols=False):
    X_target = X_target.to(torch.float64).cpu()
    X_bg     = X_bg.to(torch.float64).cpu()

    # 1) Background PCA
    Xb_c = X_bg - X_bg.mean(0, keepdim=True) # center bg by bg mean
    U, S, Vt = torch.linalg.svd(Xb_c, full_matrices=False)   # [N_b,r], [r], [r,H] # S = singular values, Vt = principal directions
    var_b_all = (S**2) / max(1, X_bg.shape[0]-1)             # [r] # convert singular values --> PC variances
    Vb_all = Vt.T                                            # [H, r] # each column in the transpose is a principal component

    # 2) Truncate to keep 'keep_var' fraction of variance
    
    order = torch.argsort(var_b_all, descending=True)       #sort PCs by variance (done by convention in trad PCA)
    var_b_sorted = var_b_all[order]
    Vb_sorted = Vb_all[:, order]
    cum = torch.cumsum(var_b_sorted, dim=0) / var_b_sorted.sum()                           # cumulative sum of variance along sorted PCs
    r_keep = int(torch.searchsorted(cum, torch.tensor(keep_var), right=True).item()) + 1   # ID num PCs to keep to meet keep_var (cut off PCs that don't contribute to explained variance, helps with stability during whitening)
    var_b = var_b_sorted[:r_keep].contiguous()
    Vb    = Vb_sorted[:, :r_keep].contiguous()

    # 3) Floor small eigenvalues before inverting
    # whitening involves multiplying by the inv_sqrt(var) to normalize background dist
    # this step sets a floor for these var, so that the inv_sqrt doesn't explode
    tau = eig_floor_frac * var_b.mean()
    inv_sqrt = (var_b.clamp_min(tau) + eps).rsqrt()

    # 4) Whitening
    mu_b = X_bg.mean(0, keepdim=True)
    def whiten(X):
        Xc = X - mu_b                # center on background mean
        return (Xc @ Vb) * inv_sqrt  # [N, r_keep] project onto kept PCs and scale so that background data has unit variance on *each* PC

    Zt = whiten(X_target)    # whiten target data
    # (Optional) Zt = Zt - Zt.mean(0, keepdim=True)

    # 5) PCA on whitened target
    Uz, Sz, Vzt = torch.linalg.svd(Zt, full_matrices=False)   # again, S = singluar values, Vt = principal directions
    lam = (Sz**2) / max(1, Zt.shape[0]-1)   # "explained variance" in whitened space equivalent to rayleigh ratios
                                            # rayleigh ratio = var(target)/var(background) along each axis
                                            # this is equivalent here, since var(background) = 1 after whitening
    Vz  = Vzt.T                             # [r_keep, r_z] # transpose to get principal components

    # 6) Back-map + optional column renorm to unit bg variance
    C = (Vb * inv_sqrt.unsqueeze(0)) @ Vz   # [H, r_z] # un-whiten PCs to get them into original space

    if renorm_cols: # normalize the components *again* to ensure normal dist for bg data
                    # also helps with interpretability: projection of +1 = "one background sigma above mean"
                    # stabilizes rayleigh ratios b/c orthonormal scaling restored
        Xb0 = X_bg - mu_b
        # exact unit variance on harmless along every component
        s2 = (Xb0 @ C).var(dim=0, unbiased=True)  # [r_z]
        C = C / torch.sqrt(s2.clamp_min(1e-12))   # rescale columns

    lam_pos = lam.clamp_min(0)                         # clean up possible negatives here
    explained = lam_pos / lam_pos.sum().clamp_min(eps) # normalize variance

    return {"mu_b": mu_b.squeeze(0),     # background mean
            "V_b": Vb, "var_b": var_b,   # whitening PCs and variances
            "Vz": Vz, "lam": lam,        # cPCs and variances in whitened space
            "components": C,             # cPCs in original space
            "explained": explained}      # cPC variances in original space

# iterate through layers and token positions, choose "best" PCA based on contrastive variance concentration

@torch.no_grad()
def get_lp_slices(X_h_lpnh, X_s_lpnh, layer, pos):
    Xt = X_h_lpnh[layer, pos].to(torch.float64, non_blocking=True).cpu()
    Xb = X_s_lpnh[layer, pos].to(torch.float64, non_blocking=True).cpu()
    return Xt, Xb

def rayleigh_from_whitened(lam):
    # lam is from PCA on Zt; equals Rayleigh ratios.
    return float(lam[0].item())

@torch.no_grad()
def mean_gap_on_component(c, Xt, Xb, mu_b):
    # project *centered-by-mu_b* data onto the back-mapped component
    zt = (Xt - mu_b) @ c
    zb = (Xb - mu_b) @ c
    gap = (zt.mean() - zb.mean()).abs().item()
    denom = (zb.var(unbiased=True).sqrt().item() + 1e-12)  # background scale
    return gap / denom

def alt_score_white(res, Xt, Xb):              
    c1     = res["components"][:, 0]         # dirst component in original space
    rr     = rayleigh_from_whitened(res["lam"])
    mg     = mean_gap_on_component(c1, Xt, Xb, res["mu_b"])
    score  = rr + 0.1*mg
    return score

@torch.no_grad()
def align_sign_by_means(C, Xt, Xb, mu_b):
    """
    C: (H,k) cPCA components (raw, not renormed preferred for steering)
    Xt, Xb: (N,H) activations at the SAME (layer,pos) used for cPCA
    mu_b: (H,) background mean from cpca_whitened
    Returns C_signed, s where s is (k,) in {+1,-1}
    """
    # center by background mean (matches how C was derived)
    Ht = (X_harm - mu_b) @ C     # (N_h, k)
    Hb = (X_harml - mu_b) @ C    # (N_r, k)

    diff = Ht.mean(dim=0) - Hb.mean(dim=0)   # (k,)
    s = torch.where(diff >= 0, 1.0, -1.0)    # choose sign so harmful > harmless
    C_signed = C * s  # broadcast over rows

    return C_signed, s


def choose_best_cpca(X_h, X_s, topk: int = 3, keep_var = 0.999, eig_floor_frac = 1e-3):
    L, P = X_h.shape[:2]
    best = (-1.0, None, None, None)  # (score, li, pi, result)
    for li in range(L):
        print(f"Layer{li}")
        for pi in range(P):
            Xt, Xb = get_lp_slices(X_h, X_s, li, pi)
            res = cpca_whitened(Xt, Xb, keep_var = keep_var, eig_floor_frac = eig_floor_frac)
            score = alt_score_white(res, Xt, Xb)
            res["components"], _ = align_sign_by_means(res["components"], Xt, Xb, res["mu_b"])
            
            if score > best[0]:
                best = (score, li, pi, res)
    _, li, pi, res = best
    
    return li, pi, res  # res contains components, lam, explained_contrastive, etc.
