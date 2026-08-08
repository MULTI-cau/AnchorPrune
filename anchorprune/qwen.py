"""Qwen2.5-VL-specific integration for AnchorPrune."""

from __future__ import annotations

from types import MethodType
from typing import Callable

import torch
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(v)


def _sanitize_query(text: str) -> str:
    return ("" if text is None else str(text)).strip()


def _extract_query_from_doc_record(doc) -> str:
    if not isinstance(doc, dict):
        return ""

    for k in ("question", "query", "prompt", "instruction", "problem", "text", "caption"):
        v = doc.get(k, None)
        q = _sanitize_query(v)
        if q:
            return q

    conv = doc.get("conversations", None)
    if isinstance(conv, list):
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            src = str(turn.get("from", "")).lower()
            if src in {"human", "user"}:
                q = _sanitize_query(turn.get("value", ""))
                if q:
                    return q
    return ""


def _extract_task_query(contexts, batched_messages, task_dict=None, task=None, split=None, doc_id=None) -> str:
    if contexts and len(contexts) > 0:
        q = _sanitize_query(contexts[0]).replace("<image>", "").strip()
        if q:
            return q

    if batched_messages and len(batched_messages) > 0:
        msg0 = batched_messages[0]
        if isinstance(msg0, list):
            for turn in msg0:
                if turn.get("role") != "user":
                    continue
                content = turn.get("content", [])
                if isinstance(content, list):
                    chunks = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            t = _sanitize_query(part.get("text", ""))
                            if t:
                                chunks.append(t)
                    q = _sanitize_query(" ".join(chunks))
                    if q:
                        return q

    try:
        if task_dict is not None and task is not None and split is not None and doc_id is not None and len(doc_id) > 0:
            doc0 = task_dict[task][split][doc_id[0]]
            q = _extract_query_from_doc_record(doc0)
            if q:
                return q
    except Exception:
        pass

    return ""


@torch.no_grad()
def _igfps_continue_with_presel(v: torch.Tensor, w: torch.Tensor, k_total: int, preselected: torch.Tensor) -> torch.Tensor:
    device = v.device
    n = int(v.shape[0])
    k_total = min(int(k_total), n)
    if k_total <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    vn = F.normalize(v.float(), dim=-1)
    w = torch.clamp(w.float(), min=0.0)

    selected_mask = torch.zeros((n,), dtype=torch.bool, device=device)
    selected_idx = torch.empty((k_total,), dtype=torch.long, device=device)
    selected_count = 0

    if preselected is not None and preselected.numel() > 0:
        pre = preselected.to(device=device, dtype=torch.long).reshape(-1)
        pre = pre[(pre >= 0) & (pre < n)]
        if pre.numel() > 0:
            pre = torch.unique(pre, sorted=False)
        if pre.numel() > 0:
            keep = min(int(pre.numel()), k_total)
            selected_idx[:keep] = pre[:keep]
            selected_mask.scatter_(0, pre[:keep], True)
            selected_count = keep

    if selected_count == 0:
        first = int(w.argmax().item())
        selected_idx[0] = first
        selected_mask[first] = True
        selected_count = 1

    min_dist = torch.ones((n,), dtype=torch.float32, device=device)
    if selected_count > 0:
        pre = selected_idx[:selected_count]
        dist_to_pre = 1.0 - (vn @ vn.index_select(0, pre).t())
        min_dist = torch.minimum(min_dist, dist_to_pre.min(dim=1).values)
        min_dist[selected_mask] = -1e9

    while selected_count < k_total:
        score = min_dist * w
        score[selected_mask] = -1e9
        nxt = int(score.argmax().item())
        selected_idx[selected_count] = nxt
        selected_mask[nxt] = True
        selected_count += 1

        dist_to_nxt = 1.0 - (vn @ vn[nxt])
        min_dist = torch.minimum(min_dist, dist_to_nxt)
        min_dist[nxt] = -1e9

    return selected_idx[:selected_count]


def patch_qwen_model_for_pruning(wrapper) -> None:
    """Install the AnchorPrune runtime hook on a compatible Qwen2.5-VL wrapper."""
    model = wrapper._model
    core = model.model
    if bool(getattr(core, "_qwen_prune_runtime_patched", False)):
        return

    if not hasattr(model, "_qwen_generate_orig"):
        model._qwen_generate_orig = model.generate

    def _generate_patched(self_model, *args, **kwargs):
        cfg = self_model.config
        if bool(getattr(cfg, "qwen_prune_enable", False)):
            qtxt = _sanitize_query(getattr(cfg, "qwen_prune_query_text", ""))
            if len(qtxt) == 0:
                ids = kwargs.get("input_ids", None)
                if ids is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
                    ids = args[0]
                if isinstance(ids, torch.Tensor) and ids.numel() > 0:
                    try:
                        decoded = wrapper.tokenizer.decode(ids[0], skip_special_tokens=True)
                        decoded = _sanitize_query(decoded)
                        cfg.qwen_prune_query_text = decoded
                    except Exception:
                        pass
        return self_model._qwen_generate_orig(*args, **kwargs)

    model.generate = MethodType(_generate_patched, model)

    if not hasattr(core, "_qwen_get_image_features_orig"):
        core._qwen_get_image_features_orig = core.get_image_features

    # Estimate the global importance prior from received attention mass, matching p_i = (1 / HN) sum_h sum_j A_{h,j,i}.
    if not bool(getattr(core, "_qwen_prune_attn_mass_patched", False)):
        visual = core.visual
        full_layers = [int(x) for x in getattr(visual, "fullatt_block_indexes", [])]
        target_layer = max(full_layers) if len(full_layers) > 0 else (len(visual.blocks) - 1)
        requested_layer = int(getattr(core.config, "qwen_prune_stage2_layer", -1))
        if 0 <= requested_layer < len(visual.blocks):
            core.config.qwen_prune_stage2_layer = int(requested_layer)
        else:
            core.config.qwen_prune_stage2_layer = int(target_layer)

        def _make_attn_forward_with_mass(orig_forward, layer_idx: int):
            @torch.no_grad()
            def _forward_with_mass(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
                need_mass = bool(getattr(core.config, "qwen_prune_enable", False)) and bool(
                    getattr(core.config, "qwen_prune_stage2_use_attn_mass", True)
                )
                if (not need_mass) or (int(layer_idx) != int(getattr(core.config, "qwen_prune_stage2_layer", layer_idx))):
                    return orig_forward(
                        hidden_states,
                        cu_seqlens=cu_seqlens,
                        rotary_pos_emb=rotary_pos_emb,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )

                seq_length = hidden_states.shape[0]
                query_states, key_states, value_states = (
                    self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
                )
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

                query_states = query_states.transpose(0, 1).unsqueeze(0)
                key_states = key_states.transpose(0, 1).unsqueeze(0)
                value_states = value_states.transpose(0, 1).unsqueeze(0)

                attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                    self.config._attn_implementation, eager_attention_forward
                )

                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                q_splits = torch.split(query_states, lengths.tolist(), dim=2)
                k_splits = torch.split(key_states, lengths.tolist(), dim=2)
                v_splits = torch.split(value_states, lengths.tolist(), dim=2)

                recv_mass_chunks = []
                attn_outputs = []
                chunk_q = 64
                for q, k, v in zip(q_splits, k_splits, v_splits):
                    out = attention_interface(
                        self,
                        q,
                        k,
                        v,
                        attention_mask=None,
                        scaling=self.scaling,
                        dropout=0.0 if not self.training else self.attention_dropout,
                        is_causal=False,
                        **kwargs,
                    )[0]
                    attn_outputs.append(out)

                    qf = q.float()
                    kf = k.float()
                    n_local = int(qf.shape[2])
                    recv_sum = torch.zeros((int(qf.shape[1]), n_local), device=qf.device, dtype=torch.float32)
                    for s in range(0, n_local, chunk_q):
                        e = min(n_local, s + chunk_q)
                        scores = torch.matmul(qf[:, :, s:e, :], kf.transpose(2, 3)) * float(self.scaling)
                        attn = torch.softmax(scores, dim=-1)
                        recv_sum += attn.squeeze(0).sum(dim=1)
                    recv_mass = (recv_sum / float(n_local)).mean(dim=0)
                    recv_mass_chunks.append(recv_mass)

                self._qwen_last_received_mass = torch.cat(recv_mass_chunks, dim=0)

                attn_output = torch.cat(attn_outputs, dim=1)
                attn_output = attn_output.reshape(seq_length, -1).contiguous()
                attn_output = self.proj(attn_output)
                return attn_output

            return _forward_with_mass

        for li, blk in enumerate(visual.blocks):
            if not hasattr(blk.attn, "_qwen_attn_forward_orig"):
                blk.attn._qwen_attn_forward_orig = blk.attn.forward
            blk.attn.forward = MethodType(_make_attn_forward_with_mass(blk.attn._qwen_attn_forward_orig, li), blk.attn)

        core._qwen_prune_attn_mass_patched = True

    @torch.no_grad()
    def _get_image_features_patched(self, pixel_values, image_grid_thw=None, **kwargs):
        outs = self._qwen_get_image_features_orig(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)
        cfg = self.config
        full_sizes = [int(x.shape[0]) for x in getattr(outs, "pooler_output", [])]
        cfg.qwen_prune_last_full_tokens_batch = full_sizes
        if not bool(getattr(cfg, "qwen_prune_enable", False)):
            return outs
        if image_grid_thw is None:
            return outs

        embeds_list = list(outs.pooler_output)
        if len(embeds_list) == 0:
            return outs

        query_text = _sanitize_query(getattr(cfg, "qwen_prune_query_text", ""))
        if len(query_text) == 0:
            return outs

        tok = wrapper.tokenizer(query_text, return_tensors="pt", add_special_tokens=False)
        q_ids = tok.input_ids.to(embeds_list[0].device)
        q_tok = self.get_input_embeddings()(q_ids)[0].float()
        if q_tok.numel() == 0:
            return outs
        q_tok = F.normalize(q_tok, dim=-1)

        raw_hidden = getattr(outs, "last_hidden_state", None)
        raw_splits = None
        raw_sizes = image_grid_thw.prod(-1).tolist()
        if raw_hidden is not None:
            if sum(raw_sizes) == int(raw_hidden.shape[0]):
                raw_splits = list(torch.split(raw_hidden.float(), raw_sizes, dim=0))

        attn_mass_splits = None
        reverse_splits = None
        use_attn_mass = bool(getattr(cfg, "qwen_prune_stage2_use_attn_mass", True))
        if use_attn_mass:
            layer_idx = int(getattr(cfg, "qwen_prune_stage2_layer", -1))
            if 0 <= layer_idx < len(self.visual.blocks):
                attn_mass_all = getattr(self.visual.blocks[layer_idx].attn, "_qwen_last_received_mass", None)
                if attn_mass_all is not None and int(attn_mass_all.numel()) == int(sum(raw_sizes)):
                    attn_mass_splits = list(torch.split(attn_mass_all.float(), raw_sizes, dim=0))
                    try:
                        window_index, _ = self.visual.get_window_index(image_grid_thw)
                        if not torch.is_tensor(window_index):
                            window_index = torch.tensor(window_index, device=attn_mass_all.device, dtype=torch.long)
                        else:
                            window_index = window_index.to(device=attn_mass_all.device, dtype=torch.long)
                        reverse_indices = torch.argsort(window_index)
                        merge_unit = int(getattr(self.visual, "spatial_merge_unit", 4))
                        merge_sizes = [int(s // merge_unit) for s in raw_sizes]
                        if int(reverse_indices.numel()) == int(sum(merge_sizes)):
                            reverse_splits = list(torch.split(reverse_indices, merge_sizes, dim=0))
                    except Exception:
                        reverse_splits = None

        k_total_cfg = max(0, int(getattr(cfg, "qwen_prune_k_total", 0)))
        fixed_k_rel = int(getattr(cfg, "qwen_prune_k_rel", -1))
        adaptive_rel = bool(getattr(cfg, "qwen_prune_adaptive_rel", True))
        tau = float(getattr(cfg, "qwen_prune_tau", 0.2))
        patience = max(1, int(getattr(cfg, "qwen_prune_patience", 3)))
        patience_mode = str(getattr(cfg, "qwen_prune_patience_mode", "cumulative")).strip().lower()
        anchor_k_cfg = max(1, int(getattr(cfg, "qwen_prune_anchor_k", 5)))
        kmin_abs = max(1, int(getattr(cfg, "qwen_prune_kmin_abs", anchor_k_cfg)))
        kmax_ratio = float(getattr(cfg, "qwen_prune_kmax_ratio", 0.5))

        selected_features = []
        k_rel_batch = []
        for b, visual_features in enumerate(embeds_list):
            n = int(visual_features.shape[0])
            if n <= 0:
                selected_features.append(visual_features)
                k_rel_batch.append(0)
                continue

            k_total = min(k_total_cfg if k_total_cfg > 0 else n, n)
            if k_total <= 0:
                selected_features.append(visual_features[:0])
                k_rel_batch.append(0)
                continue

            visual_features_float = visual_features.float()
            visual_features_norm = F.normalize(visual_features_float, dim=-1)

            sim = visual_features_norm @ q_tok.t()
            sim_agg = str(getattr(cfg, "qwen_prune_sim_agg", "max")).strip().lower()
            if sim_agg in {"max", "tokenmax", "token_wise_max", "token-wise-max"}:
                rel_score = sim.max(dim=-1).values
            elif sim_agg in {"mean_embed", "embed_mean", "text_mean", "mean_text", "mean-embed", "text-mean"}:
                q_mean = F.normalize(q_tok.mean(dim=0, keepdim=True), dim=-1)[0]
                rel_score = visual_features_norm @ q_mean
            else:
                rel_score = sim.mean(dim=-1)
            rel_sorted = torch.argsort(rel_score, descending=True)

            k_min_eff = min(kmin_abs, k_total, n)
            k_max_eff = max(k_min_eff, int(kmax_ratio * k_total))
            k_max_eff = min(k_max_eff, k_total, n)

            chosen_t = k_max_eff
            if (not adaptive_rel) and fixed_k_rel >= 0:
                chosen_t = min(fixed_k_rel, k_total, n)
            elif adaptive_rel and (k_max_eff > k_min_eff):
                high_novelty_streak = 0
                high_novelty_total = 0
                anchor_k = min(anchor_k_cfg, int(rel_sorted.numel()))
                anchor_idx = rel_sorted[:anchor_k]
                for t in range(k_min_eff + 1, k_max_eff + 1):
                    cur_idx = rel_sorted[t - 1]
                    prev_idx_eff = anchor_idx[anchor_idx != cur_idx]
                    if prev_idx_eff.numel() == 0:
                        delta_t = 0.0
                    else:
                        delta_t = (
                            1.0
                            - (
                                visual_features_norm[cur_idx]
                                @ visual_features_norm.index_select(0, prev_idx_eff).t()
                            )
                        ).min().item()
                    is_high_novelty = delta_t > tau
                    if patience_mode in {"cumulative", "total"}:
                        if is_high_novelty:
                            high_novelty_total += 1
                        if high_novelty_total >= patience:
                            chosen_t = t
                            break
                    else:
                        if is_high_novelty:
                            high_novelty_streak += 1
                        else:
                            high_novelty_streak = 0
                        if high_novelty_streak >= patience:
                            chosen_t = t
                            break

            rel_idx = rel_sorted[:chosen_t]
            k_rel_batch.append(int(rel_idx.numel()))

            if (attn_mass_splits is not None) and (b < len(attn_mass_splits)):
                raw_m = attn_mass_splits[b]
                if raw_m.numel() >= 4 * n and (raw_m.numel() % 4 == 0):
                    raw_groups = raw_m[: 4 * n].view(n, 4)
                    merge_mode = str(getattr(cfg, "qwen_prune_stage2_merge", "mean")).strip().lower()
                    if merge_mode == "max":
                        w_merge = raw_groups.max(dim=1).values
                    else:
                        w_merge = raw_groups.mean(dim=1)
                elif raw_m.numel() == n:
                    w_merge = raw_m
                else:
                    idx = torch.linspace(0, max(0, raw_m.numel() - 1), steps=n, device=raw_m.device).long()
                    w_merge = raw_m[idx]
                if (reverse_splits is not None) and (b < len(reverse_splits)):
                    ridx = reverse_splits[b].to(device=w_merge.device, dtype=torch.long).reshape(-1)
                    ridx_valid = (ridx >= 0) & (ridx < n)
                    if int(ridx.numel()) == int(n) and bool(torch.all(ridx_valid)):
                        w_merge = w_merge.index_select(0, ridx)
                w_merge = torch.clamp(w_merge, min=0.0)
            elif raw_splits is not None and b < len(raw_splits):
                raw_b = raw_splits[b]
                raw_b = F.normalize(raw_b, dim=-1)
                gap_head = F.normalize(raw_b.mean(dim=0, keepdim=True), dim=-1)[0]
                w_raw = (raw_b @ gap_head).clamp(min=0.0)

                if w_raw.numel() >= 4 * n and (w_raw.numel() % 4 == 0):
                    raw_groups = w_raw[: 4 * n].view(n, 4)
                    merge_mode = str(getattr(cfg, "qwen_prune_stage2_merge", "mean")).strip().lower()
                    if merge_mode == "max":
                        w_merge = raw_groups.max(dim=1).values
                    else:
                        w_merge = raw_groups.mean(dim=1)
                elif w_raw.numel() == n:
                    w_merge = w_raw
                else:
                    idx = torch.linspace(0, max(0, w_raw.numel() - 1), steps=n, device=w_raw.device).long()
                    w_merge = w_raw[idx]
            else:
                gap_head = F.normalize(visual_features_float.mean(dim=0, keepdim=True), dim=-1)[0]
                w_merge = (visual_features_norm @ gap_head).clamp(min=0.0)

            if bool(getattr(cfg, "qwen_prune_stage2_minmax_norm", False)):
                w_min = w_merge.min()
                w_max = w_merge.max()
                denom = w_max - w_min
                if torch.isfinite(denom) and (denom > 0):
                    w_merge = (w_merge - w_min) / denom
                else:
                    w_merge = torch.zeros_like(w_merge)
            elif bool(getattr(cfg, "qwen_prune_stage2_l1_norm", False)):
                w_sum = w_merge.sum()
                if torch.isfinite(w_sum) and (w_sum > 0):
                    w_merge = w_merge / w_sum

            if k_total > int(rel_idx.numel()):
                selected_indices = _igfps_continue_with_presel(
                    v=visual_features_float,
                    w=w_merge.float(),
                    k_total=k_total,
                    preselected=rel_idx,
                )
            else:
                selected_indices = rel_idx[:k_total]

            selected_indices = torch.sort(selected_indices).values
            selected_indices = selected_indices[(selected_indices >= 0) & (selected_indices < n)]
            if selected_indices.numel() == 0:
                selected_indices = rel_idx[: min(k_total, int(rel_idx.numel()))]
            selected_features.append(visual_features.index_select(0, selected_indices))

        cfg.qwen_prune_last_k_rel_batch = k_rel_batch
        cfg.qwen_prune_last_selected_tokens_batch = [int(x.shape[0]) for x in selected_features]
        outs.pooler_output = tuple(selected_features)
        return outs

    core.get_image_features = MethodType(_get_image_features_patched, core)
    core._qwen_prune_runtime_patched = True
