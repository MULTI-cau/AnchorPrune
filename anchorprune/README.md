# AnchorPrune Core

This directory contains the core implementation of **AnchorPrune: Relevance-Anchored Contextual Expansion for Visual Token Pruning**.

The code is intentionally separated from the LMMs-Eval wrappers. Model-specific adapters extract the token-level signals required by the method, while `selection.py` implements the model-agnostic selector used by all supported backbones.

## Modules

| File | Description |
| --- | --- |
| `selection.py` | Relevance-anchored two-stage token selection. |
| `llava.py` | LLaVA integration using negated CLIP patch-text similarity and `[CLS]`-attention importance. |
| `qwen.py` | Qwen integration using post-projector token matching and received-attention importance. |
| `__init__.py` | Public exports for direct selector use. |

## Selector Interface

`anchorprune_select` expects three token-level inputs:

- `relevance`: query-conditioned anchoring priority, shape `[N]`;
- `features`: token features for adaptive Stage-1 novelty, shape `[N, D]`;
- `importance`: Stage-2 global importance prior `p_i`, shape `[N]`.

It returns the final retained-token indices and the protected Stage-1 anchor indices. Retained tokens are restored to their native visual-token order before language-model inference.

```python
from anchorprune import AnchorPruneConfig, anchorprune_select

selected_indices, anchor_indices = anchorprune_select(
    relevance=relevance_scores,
    features=stage1_features,
    importance=importance_prior,
    config=AnchorPruneConfig(k_total=64, k_min=10),
    expansion_features=stage2_features,
)
```

The released adapters implement the paper configuration: adaptive protected anchoring, cumulative novelty stopping, `K_max = 0.5K`, and Stage-2 expansion by maximizing `p_i · Delta(i; S)`. All integrations are inference-time only and leave the underlying model parameters unchanged.
