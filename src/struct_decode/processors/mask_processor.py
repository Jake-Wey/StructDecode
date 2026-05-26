"""Token mask processor for simple constrained decoding."""

from typing import Optional, Set

import torch

from struct_decode.processors.base import BaseLogitsProcessor

class TokenMaskProcessor(BaseLogitsProcessor):
    """
    A simple logits processor that masks out tokens not in the allowed set.

    Example:
        >>> processor = TokenMaskProcessor(allowed_token_ids={1, 2, 3})
        >>> # Model can now only generate tokens 1, 2, or 3
    """

    def __init__(
        self,
        allowed_token_ids: Optional[Set[int]] = None,
        vocab_size: Optional[int] = None
    ):
        """
        Initialize the token mask processor.

        Args:
            allowed_token_ids: Set of allowed token IDs. If None, all tokens are allowed.
            vocab_size: Size of the vocabulary. Required for creating the mask tensor.
        """

        self._allowed_token_ids = allowed_token_ids
        self._vocab_size = vocab_size
        self._mask: Optional[torch.Tensor] = None  


    def set_allowed_tokens(self, allowed_token_ids: Set[int]) -> None:
        """Update the set of allowed tokens."""

        self._allowed_token_ids = allowed_token_ids
        self._mask = None

    def _build_mask(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        """Build the mask tensor for the given vocab size."""
        
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        if self._allowed_token_ids is not None:
            mask[:] = True
            for token_id in self._allowed_token_ids:
                if 0 <= token_id < vocab_size:
                    mask[token_id] = False
        
        return mask

    def __call__(
        self, 
        input_ids: torch.Tensor,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """Apply the token mask to the logits."""
        
        vocab_size = scores.shape[-1]

        if self._mask is None or self._mask.shape[0] != vocab_size:
            self._mask = self._build_mask(vocab_size, scores.device)

        if self._mask.device != scores.device:
            self._mask = self._mask.to(scores.device)

        scores = scores.masked_fill(self._mask, float("-inf"))

        return scores
    
    def reset(self) -> None:
        """Reset the processor state."""

        self._mask = None
