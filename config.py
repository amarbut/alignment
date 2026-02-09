
import os

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class Config:
    model_alias: str
    model_path: str
    n_train: int = 128
    n_test: int = 100
    n_val: int = 32
    filter_train: bool = True
    filter_val: bool = True
    evaluation_datasets: Tuple[str] = ("jailbreakbench",)
    max_new_tokens: int = 100
    jailbreak_eval_methodologies: Tuple[str] = ("substring_matching", "openai")
    refusal_eval_methodologies: Tuple[str] = ("substring_matching", "openai") #llamaguard doesn't make sense, since it detects unsafe responses, not response vs. refusal
    ce_loss_batch_size: int = 2
    ce_loss_n_batches: int = 2048

    # Filter thresholds (used by select_direction and expert_selection)
    kl_threshold: float = 0.1
    steering_score_threshold: float = 0.0
    prune_layer_percentage: float = 0.2

    # System prompt option
    system_prompt: str = "lightweight"  # "none", "llama_2", "lightweight"

    # Intervention parameters
    coeff: float = 1.0
    threshold: float = 15.0
    num_interventions: int = 1

    def artifact_path(self) -> str:
        """Return base artifact path (backwards compatible)."""
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", self.model_alias)

    def artifact_path_with_hyperparams(self) -> str:
        """Return artifact path with hyperparameters encoded in folder name."""
        parts = [
            os.path.dirname(os.path.realpath(__file__)),
            "runs",
            self.model_alias,
            f"coeff{self.coeff}_thresh{self.threshold}_sys{self.system_prompt}_n{self.num_interventions}"
        ]
        return os.path.join(*parts)

