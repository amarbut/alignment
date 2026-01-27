"""
Base class for jailbreak attack methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class JailbreakMethod(ABC):
    """
    Abstract base class for black-box jailbreak attack methods.

    Each method transforms a harmful prompt into an adversarial prompt
    designed to bypass safety mechanisms.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def transform(self, prompt: str) -> str:
        """
        Transform a prompt into an adversarial version.

        Args:
            prompt: The original harmful prompt

        Returns:
            The transformed adversarial prompt
        """
        pass

    def transform_batch(self, prompts: List[str]) -> List[str]:
        """
        Transform a batch of prompts.

        Args:
            prompts: List of original prompts

        Returns:
            List of transformed prompts
        """
        return [self.transform(p) for p in prompts]

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata about this attack method.
        """
        return {
            "name": self.name,
            "type": "black-box",
        }
