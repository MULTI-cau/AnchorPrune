"""Public AnchorPrune selector interface.

The exported utilities are intentionally independent of the LMMs-Eval
evaluation harness. Model-specific adapters provide relevance, feature, and
importance signals, while the selector implements the training-free pruning
procedure described in the paper.
"""

from .selection import (
    AnchorPruneConfig,
    adaptive_relevance_anchor,
    importance_weighted_expansion,
    anchorprune_select,
)

__all__ = [
    "AnchorPruneConfig",
    "adaptive_relevance_anchor",
    "importance_weighted_expansion",
    "anchorprune_select",
]
