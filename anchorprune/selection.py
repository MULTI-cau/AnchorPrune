"""Model-agnostic AnchorPrune selector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AnchorPruneConfig:
    """
    Configuration for AnchorPrune's relevance-anchored contextual expansion selector.

    Args:
        k_total: Final retained visual-token budget 'K'.
        k_min: Minimum protected relevance-anchor size 'K_min'.
        tau: Novelty threshold for adaptive anchor expansion.
        patience: Number of high-novelty relevance-ranked candidates required before stopping anchor expansion.  The release uses cumulative counting.
        kmax_ratio: Upper bound ratio for Stage 1.
    """

    k_total: int
    k_min: int
    tau: float = 0.2
    patience: int = 3
    kmax_ratio: float = 0.5


def _as_1d(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim != 1:
        raise ValueError(
            f"AnchorPrune expects `{name}` to be a one-dimensional token score tensor, but received shape {tuple(x.shape)}."
        )
    return x


def _as_2d(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(
            f"AnchorPrune expects `{name}` to be a two-dimensional token feature tensor, but received shape {tuple(x.shape)}."
        )
    return x


def cosine_novelty_to_set(
    candidate_features: torch.Tensor,
    anchor_features: torch.Tensor,
) -> torch.Tensor:
    """
    Compute novelty 'Delta' from each candidate to an anchor set.

    Novelty is the minimum cosine distance to the retained anchor:

    'Delta(i; S) = min_j (1 - cos(h_i, h_j))' for 'j in S'.

    Args:
        candidate_features: Candidate token features, shape '[M, D]'.
        anchor_features: Already retained token features, shape '[A, D]'.

    Returns:
        Novelty values with shape '[M]'.  If the anchor is empty, every candidate receives novelty 1.
    """

    candidate_features = _as_2d(candidate_features, "candidate_features")
    anchor_features = _as_2d(anchor_features, "anchor_features")
    if anchor_features.shape[0] == 0:
        return torch.ones(candidate_features.shape[0], device=candidate_features.device)

    cand = F.normalize(candidate_features.float(), dim=-1)
    anch = F.normalize(anchor_features.float(), dim=-1)
    dist = 1.0 - cand @ anch.t()
    return dist.min(dim=1).values


@torch.no_grad()
def adaptive_relevance_anchor(
    relevance: torch.Tensor,
    features: torch.Tensor,
    k_total: int,
    k_min: int,
    tau: float = 0.2,
    patience: int = 3,
    kmax_ratio: float = 0.5,
) -> torch.Tensor:
    """
    Select the Stage-1 protected relevance anchor.

    Tokens are sorted by query-conditioned anchoring priority, and the protected anchor is formed before contextual expansion begins. No diversity penalty is applied inside the anchor, so query-critical evidence is not displaced by redundancy reduction.

    Adaptive budget rule:
        1. Keep the first 'k_min' relevance-ranked tokens as 'A_min'.
        2. Scan later relevance-ranked tokens up to 'floor(kmax_ratio*K)'.
        3. Count candidates whose novelty against 'A_min' exceeds 'tau'.
        4. Stop at the first position where the cumulative count reaches
           'patience'; otherwise use the upper cap.

    This is the paper setting: cumulative patience, 'tau=0.2', 'patience=3', and 'kmax_ratio=0.5'.
    """

    relevance = _as_1d(relevance, "relevance")
    features = _as_2d(features, "features")
    if relevance.shape[0] != features.shape[0]:
        raise ValueError(
            f"AnchorPrune expects `relevance` and `features` to describe the same visual-token sequence, but received {relevance.shape[0]} scores and {features.shape[0]} feature vectors."
        )

    n_tokens = int(relevance.shape[0])
    if n_tokens == 0 or k_total <= 0:
        return torch.empty(0, dtype=torch.long, device=relevance.device)

    k_total = min(int(k_total), n_tokens)
    k_min_eff = min(max(1, int(k_min)), k_total)
    k_max_eff = max(k_min_eff, int(float(kmax_ratio) * k_total))
    k_max_eff = min(k_max_eff, k_total, n_tokens)
    patience = max(1, int(patience))

    ranked = torch.argsort(relevance.float(), descending=True)
    chosen = k_max_eff

    if k_max_eff > k_min_eff:
        initial_anchor = ranked[:k_min_eff]
        anchor_features = features.index_select(0, initial_anchor)
        novelty_event_count = 0

        for t in range(k_min_eff + 1, k_max_eff + 1):
            cur_idx = ranked[t - 1]
            cur_feat = features.index_select(0, cur_idx.view(1))
            delta = float(cosine_novelty_to_set(cur_feat, anchor_features)[0].item())

            if delta > tau:
                novelty_event_count += 1
            if novelty_event_count >= patience:
                chosen = t
                break

    return ranked[:chosen]


@torch.no_grad()
def importance_weighted_expansion(
    features: torch.Tensor,
    importance: torch.Tensor,
    k_total: int,
    preselected: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Run Stage-2 contextual expansion with importance-weighted novelty.

    Starting from the protected relevance anchor, greedily add tokens until 'k_total' tokens are retained. At each step the selected token maximizes:

    'p_i * Delta(i; S)'

    where 'p_i' is the global importance prior and 'Delta(i; S)' is the minimum cosine distance to the current retained set.  
    This expands the anchor with tokens that are both informative and complementary, rather than merely diverse.
    """

    features = _as_2d(features, "features")
    importance = _as_1d(importance, "importance")
    if features.shape[0] != importance.shape[0]:
        raise ValueError(
            "AnchorPrune expects `features` and `importance` to describe the same "
            f"visual-token sequence, but received {features.shape[0]} feature "
            f"vectors and {importance.shape[0]} importance scores."
        )

    device = features.device
    n_tokens = int(features.shape[0])
    if n_tokens == 0 or k_total <= 0:
        return torch.empty(0, dtype=torch.long, device=device)

    k_total = min(int(k_total), n_tokens)
    feats = F.normalize(features.float(), dim=-1)
    weights = torch.clamp(importance.float().to(device), min=0.0)

    selected_mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    selected: list[int] = []

    if preselected is not None and preselected.numel() > 0:
        pre = preselected.to(device=device, dtype=torch.long).reshape(-1)
        pre = pre[(pre >= 0) & (pre < n_tokens)]
        if pre.numel() > 0:
            pre = pre[:k_total]
            selected.extend(int(i) for i in pre.tolist())
            selected_mask.scatter_(0, pre, True)

    # If no anchor is available, initialize expansion from the strongest importance prior.
    if not selected:
        first = int(torch.argmax(weights).item())
        selected.append(first)
        selected_mask[first] = True

    selected_tensor = torch.tensor(selected, dtype=torch.long, device=device)
    min_dist = torch.ones(n_tokens, dtype=torch.float32, device=device)
    dist_to_selected = 1.0 - feats @ feats.index_select(0, selected_tensor).t()
    min_dist = torch.minimum(min_dist, dist_to_selected.min(dim=1).values)
    min_dist[selected_mask] = -1e9

    while len(selected) < k_total:
        score = min_dist * weights
        score[selected_mask] = -1e9
        nxt = int(torch.argmax(score).item())
        selected.append(nxt)
        selected_mask[nxt] = True

        dist_to_next = 1.0 - feats @ feats[nxt]
        min_dist = torch.minimum(min_dist, dist_to_next)
        min_dist[nxt] = -1e9

    # Restore the retained sequence to the original visual-token order.
    return torch.sort(torch.tensor(selected, dtype=torch.long, device=device)).values


@torch.no_grad()
def anchorprune_select(
    relevance: torch.Tensor,
    features: torch.Tensor,
    importance: torch.Tensor,
    config: AnchorPruneConfig,
    expansion_features: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run AnchorPrune relevance-anchored contextual expansion.

    Args:
        relevance: Query-conditioned anchoring-priority scores, shape '[N]'.
        features: Features used for Stage-1 anchor novelty, shape '[N, D]'.
        importance: Global importance prior 'p_i', shape '[N]'.
        config: AnchorPrune selector configuration.
        expansion_features: Feature space used to measure novelty during Stage-2 contextual expansion.

    Returns:
        '(selected_indices, relevance_anchor_indices)'.  
        'selected_indices' are sorted in original token order; 'relevance_anchor_indices' are the protected anchor before contextual expansion.
    """

    anchor = adaptive_relevance_anchor(
        relevance=relevance,
        features=features,
        k_total=config.k_total,
        k_min=config.k_min,
        tau=config.tau,
        patience=config.patience,
        kmax_ratio=config.kmax_ratio,
    )
    selected = importance_weighted_expansion(
        features=features if expansion_features is None else expansion_features,
        importance=importance,
        k_total=config.k_total,
        preselected=anchor,
    )
    return selected, anchor
