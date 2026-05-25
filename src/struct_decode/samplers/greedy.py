"""Greedy sampler implementation."""

import torch

from struct_decode.samplers.base import BaseSampler

class GreedySampler(BaseSampler):
    """
    Greedy sampling: always select the token with the highest probability.
    """

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Select the token with the highest logit value.

        Args:
            logits: Shape (batch_size, vocab_size)

        Returns:
            Token IDs with shape (batch_size,)
        """
        return torch.argmax(logits, dim=-1)
    
    @property
    def name(self) -> str:
        return "greedy"
    