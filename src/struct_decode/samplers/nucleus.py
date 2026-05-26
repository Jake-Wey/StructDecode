"""Nucleus sampler implementation."""

from typing import Optional

import torch
import torch.nn.functional as F

from struct_decode.samplers.base import BaseSampler

class NucleusSample(BaseSampler):
    """
    Nucleus sampling (also known as Top-p sampling).

    Samples from the smallest set of tokens whose cumulative probability
    exceeds p. 
    """

    def __init__(
        self,
        top_p: float = 0.9,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        min_tokens_to_keep: int = 1
    ): 
        """
        Initialize nucleus sampler.

        Args:
            top_p: Cumulative probability threshold. Default: 0.9
            temperature: Sampling temperature. Default: 1.0
            top_k: If set, only sample from top_k tokens before applying top_p.
                   Default: None (no top-k filtering)
            min_tokens_to_keep: Minimum number of tokens to keep after filtering.
                               Default: 1
        """

        if not 0.0 <= top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be > 0 or Noen, got {top_k}")
        
        self.top_p = top_p
        self.temperature = temperature
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample tokens using nucleus sampling.

        Args:
            logits: Shape (batch_size, vocab_size)

        Returns:
            Token IDs with shape (batch_size,)
        """

        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Apply top-k filtering if specified
        if self.top_k is not None:
            top_k = min(self.top_k, probs.shape[-1])
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            probs = torch.zeros_like(probs)
            probs.scatter_(dim=-1, index=top_k_indices, src=top_k_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create mask for tokens to remove
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., : self.min_tokens_to_keep] = False
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove
        )

        probs = probs.masked_fill(indices_to_remove, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return next_tokens
    
    @property
    def name(self) -> str:
        parts = [f"top_p={self.top_p}"]
        if self.temperature != 1.0:
            parts.append(f"temperature={self.temperature}")
        if self.top_k is not None:
            parts.append(f"top_k={self.top_k}")
        return f"nucleus({', '.join(parts)})"
    

class TopKSampler(BaseSampler):
    """
    Top-K sampling: sample from the K most likely tokens.
    """

    def __init__(self, top_k: int = 50, temperature: float = 1.0):
        """
        Initialize top-k sampler.

        Args:
            top_k: Number of top tokens to sample from. Default: 50
            temperature: Sampling temperature. Default: 1.0
        """

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        
        self.top_k = top_k
        self.temperature = temperature

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample from the top-k tokens.

        Args:
            logits: Shape (batch_size, vocab_size)

        Returns:
            Token IDs with shape (batch_size,)
        """
        
        if self.temperature != 1.0:
            logits = logits / self.temperature

        top_k = min(self.top_k, logits.shape[-1])
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

        top_k_probs = F.softmax(top_k_logits, dim=-1)

        sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)

        next_tokens = top_k_indices.gather(dim=-1, index=sampled_indices.unsqueeze(-1)).squeeze(-1)

        return next_tokens
    
    @property
    def name(self) -> str:
        return f"top_k(k={self.top_k}, temperature={self.temperature})"