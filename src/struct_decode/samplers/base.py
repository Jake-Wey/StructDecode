"""Base class for samplers."""

from abc import ABC, abstractmethod

import torch

class BaseSampler(ABC):
    """
    Abstract base class for token samplers.
    """

    @abstractmethod
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample the next token from the logits distribution.

        Args:
            logits: The processed logits. Shape: (batch_size, vocab_size)

        Returns:
            The selected token IDs. Shape: (batch_size,)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the sampler."""
        pass