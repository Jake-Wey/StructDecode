"""Logits processors for constrained decoding."""

from struct_decode.processors.base import BaseLogitsProcessor
from struct_decode.processors.mask_processor import TokenMaskProcessor

__all__ = ["BaseLogitsProcessor", "TokenMaskProcessor"]