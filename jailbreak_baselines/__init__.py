"""
Jailbreaking baselines for comparison with MoE steering methods.

Black-box methods (no gradient access required):
- ArtPrompt (Jiang et al., 2024): ASCII art word masking
- DeepInception (Li et al., 2023): Nested fictional scenarios
- FFA (Zhou et al., 2024): Fallacy Failure Attack

White-box methods (gradient access required):
- GCG (Zou et al., 2023): Greedy Coordinate Gradient

References:
- ArtPrompt: https://arxiv.org/abs/2402.11753
- DeepInception: https://arxiv.org/abs/2311.03191
- FFA: https://arxiv.org/abs/2407.00869
- GCG: https://arxiv.org/abs/2307.15043
"""

from .base import JailbreakMethod
from .artprompt import ArtPromptAttack
from .deep_inception import DeepInceptionAttack
from .ffa import FFAAttack
from .gcg import GCGAttack, GCGConfig, run_gcg_attack

__all__ = [
    "JailbreakMethod",
    "ArtPromptAttack",
    "DeepInceptionAttack",
    "FFAAttack",
    "GCGAttack",
    "GCGConfig",
    "run_gcg_attack",
]
