"""Base class for logits processors."""

from abc import ABC, abstractmethod
from typing import List

import torch

class BaseLogitsProcessor(ABC):
    """
    Abstract base class for all logits processors.
    """
        
    @abstractmethod
    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Process the logits for the current generation step.

        Args:
            input_ids: The input token IDs generated so far.
                       Shape: (batch_size, sequence_length)
            scores: The logits for the next token.
                    Shape: (batch_size, vocab_size)

        Returns:
            Modified logits with the same shape as input scores.
        """

        pass

    def reset(self) -> None:
        """Reset the processor state. Called at the start of generation."""

        pass

class LogitsProcessorList:
    """
    Container for multiple logits processors.
    """

    def __init__(self, processors: List[BaseLogitsProcessor]):
        self.processors = processors

    def reset(self) -> None:
        """Reset all processors."""

        for processor in self.processors:
            processor.reset()

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """Apply all processors in sequence."""
        
        for processor in self.processors:
            scores = processor(input_ids, scores)
        return scores