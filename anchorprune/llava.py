"""LLaVA-specific integration for AnchorPrune."""

from __future__ import annotations

from types import MethodType
from typing import List, Optional

import torch
import torch.nn.functional as F

from .selection import AnchorPruneConfig, anchorprune_select


def _sanitize_query(text: str) -> str:
    return ("" if text is None else str(text)).strip()


def _pick_hidden_for_layer(attentions, hidden_states, layer_idx: int) -> torch.Tensor:
    la = len(attentions)
    li = int(layer_idx)
    if li < 0:
        li = la + li
    li = max(0, min(la - 1, li))
    if len(hidden_states) == la + 1:
        return hidden_states[li + 1]
    return hidden_states[li]


def _ensure_vision_tower_pruning_helpers(vt) -> None:
    if not hasattr(vt, "text_tower"):
        vt.text_tower = None
    if not hasattr(vt, "text_tokenizer"):
        vt.text_tokenizer = None
    if not hasattr(vt, "max_position_embeddings"):
        vt.max_position_embeddings = None
    if not hasattr(vt, "_text_embeddings_cache"):
        vt._text_embeddings_cache = {}
    if not hasattr(vt, "_text_embeddings_cache_max"):
        vt._text_embeddings_cache_max = 256

    if not hasattr(vt, "forward_with_attn"):

        @torch.no_grad()
        def _forward_with_attn(self, images):
            if type(images) is list:
                raise RuntimeError("AnchorPrune requires a batched image tensor with shape [B, 3, H, W].")
            return self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )

        vt.forward_with_attn = MethodType(_forward_with_attn, vt)

    if not hasattr(vt, "load_text_tower"):

        @torch.no_grad()
        def _load_text_tower(self, device_map=None):
            if (self.text_tower is not None) and (self.text_tokenizer is not None) and hasattr(self.vision_tower, "visual_projection"):
                return

            from transformers import (
                CLIPTextModelWithProjection,
                CLIPTokenizerFast,
                CLIPVisionModelWithProjection,
            )

            vision_with_proj = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name, device_map=device_map)
            vp = vision_with_proj.visual_projection.to(device=self.device)
            vp.eval()
            for p in vp.parameters():
                p.requires_grad_(False)
            self.vision_tower.visual_projection = vp

            self.text_tokenizer = CLIPTokenizerFast.from_pretrained(self.vision_tower_name)
            self.text_tower = CLIPTextModelWithProjection.from_pretrained(self.vision_tower_name, device_map=device_map)
            self.text_tower.requires_grad_(False)
            self.text_tower.eval()
            self.text_tower = self.text_tower.to(device=self.device)
            self.max_position_embeddings = int(self.text_tower.config.max_position_embeddings)
            self._text_embeddings_cache.clear()

            del vision_with_proj
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        vt.load_text_tower = MethodType(_load_text_tower, vt)

    if not hasattr(vt, "encode_text_embeds_segments"):

        @torch.no_grad()
        def _encode_text_embeds_segments(self, text: str) -> torch.Tensor:
            self.load_text_tower(device_map=None)
            cache_key = str(text)
            cached = self._text_embeddings_cache.get(cache_key)
            if cached is not None:
                return cached

            inputs = self.text_tokenizer(text=[text], return_tensors="pt")
            max_pos = int(self.max_position_embeddings)
            seg = (inputs.input_ids.shape[1] - 1) // max_pos + 1
            pad = max_pos * seg - inputs.input_ids.shape[1]
            inputs = {
                k: torch.cat([v, v.new_zeros((v.shape[0], pad))], dim=1)
                .reshape(-1, max_pos)
                .to(device=self.device, non_blocking=True)
                for k, v in inputs.items()
            }
            out = self.text_tower(**inputs, return_dict=True)
            te = out.text_embeds.float()
            te = te.to(device=self.device, non_blocking=True)
            if len(self._text_embeddings_cache) >= self._text_embeddings_cache_max:
                self._text_embeddings_cache.pop(next(iter(self._text_embeddings_cache)))
            self._text_embeddings_cache[cache_key] = te
            return te

        vt.encode_text_embeds_segments = MethodType(_encode_text_embeds_segments, vt)

    if not hasattr(vt, "encode_patch_embeds_with_projection"):

        @torch.no_grad()
        def _encode_patch_embeds_with_projection(self, patch_tokens: torch.Tensor):
            self.load_text_tower(device_map=None)
            if not hasattr(self.vision_tower, "visual_projection"):
                raise RuntimeError("AnchorPrune requires the CLIP visual projection to compute Stage-1 anchoring priorities.")

            if patch_tokens.dim() == 2:
                x = patch_tokens.unsqueeze(0)
                single = True
            else:
                x = patch_tokens
                single = False

            x = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
            x_ln = self.vision_tower.vision_model.post_layernorm(x)
            proj_dtype = self.vision_tower.visual_projection.weight.dtype
            x_proj = self.vision_tower.visual_projection(x_ln.to(dtype=proj_dtype))
            x_proj = F.normalize(x_proj, dim=-1)
            return x_proj[0] if single else x_proj

        vt.encode_patch_embeds_with_projection = MethodType(_encode_patch_embeds_with_projection, vt)


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(v)


def patch_llava_model_for_pruning(model) -> None:
    """Install the AnchorPrune runtime hook on a compatible LLaVA model."""

    if bool(getattr(model, "_llava_prune_runtime_patched", False)):
        return

    if not hasattr(model, "get_model"):
        return

    core_model = model.get_model()
    if not hasattr(core_model, "get_vision_tower"):
        return

    vt = core_model.get_vision_tower()
    if vt is None:
        return

    _ensure_vision_tower_pruning_helpers(vt)

    @torch.no_grad()
    def _encode_images_with_anchor_expansion_pruning(self, images: torch.Tensor, image_sizes: Optional[List[List[int]]] = None) -> torch.Tensor:
        cfg = self.config
        vt_local = self.get_model().get_vision_tower()
        if vt_local is None:
            raise RuntimeError("AnchorPrune requires an initialized LLaVA vision tower.")

        if not bool(getattr(self, "_llava_text_tower_loaded", False)):
            vt_local.load_text_tower(device_map=None)
            self._llava_text_tower_loaded = True

        layer_idx = int(getattr(cfg, "llava_prune_layer_idx", -2))
        k_total = int(getattr(cfg, "llava_prune_k_total", 32))
        query_text = _sanitize_query(getattr(cfg, "llava_prune_query_text", ""))

        if images.ndim != 4:
            raise RuntimeError(f"AnchorPrune requires image tensors with shape [B, 3, H, W], but received {tuple(images.shape)}.")

        outs = vt_local.forward_with_attn(images)
        atts = outs.attentions
        hids = outs.hidden_states
        vision_hidden_states = _pick_hidden_for_layer(atts, hids, layer_idx)
        attention_tensor = atts[layer_idx if layer_idx >= 0 else len(atts) + layer_idx]

        sel_layer = int(getattr(vt_local, "select_layer", -2))
        patch_tokens_sel = hids[sel_layer][:, 1:, :]
        clip_patch_features_all = vt_local.encode_patch_embeds_with_projection(patch_tokens_sel)

        text_embeddings = vt_local.encode_text_embeds_segments(query_text)
        text_embeddings = text_embeddings.to(device=clip_patch_features_all.device, dtype=torch.float32, non_blocking=True)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        tau = float(getattr(cfg, "llava_prune_tau", 0.20))
        patience = max(1, int(getattr(cfg, "llava_prune_patience", 3)))
        anchor_min = max(1, int(getattr(cfg, "llava_prune_anchor_k", 5)))
        kmax_ratio = float(getattr(cfg, "llava_prune_kmax_ratio", 0.5))

        def _normalized_anchor_priority(clip_patch_features: torch.Tensor) -> torch.Tensor:
            raw_clip_similarity = clip_patch_features @ text_embeddings.t()
            anchor_priority = (-raw_clip_similarity).mean(dim=-1)
            priority_min = anchor_priority.min()
            priority_max = anchor_priority.max()
            return (anchor_priority - priority_min + 1e-6) / (priority_max - priority_min + 1e-12)

        selected_feature_batches = []
        for b in range(images.shape[0]):
            vision_features = vision_hidden_states[b, 1:, :].float()
            importance_prior = attention_tensor[b, :, 0, 1:].mean(dim=0).float()
            importance_prior = importance_prior / importance_prior.sum().clamp_min(1e-12)
            clip_patch_features = clip_patch_features_all[b].float()
            normalized_priority = _normalized_anchor_priority(clip_patch_features)

            num_tokens = int(vision_features.shape[0])
            if k_total <= 0 or num_tokens <= 0:
                selected_indices = torch.empty((0,), dtype=torch.long, device=vision_features.device)
            else:
                selected_indices, _ = anchorprune_select(
                    relevance=normalized_priority,
                    features=clip_patch_features,
                    importance=importance_prior,
                    config=AnchorPruneConfig(
                        k_total=min(k_total, num_tokens),
                        k_min=anchor_min,
                        tau=tau,
                        patience=patience,
                        kmax_ratio=kmax_ratio,
                    ),
                    expansion_features=vision_features,
                )

            selected_feature_batches.append(vision_features.index_select(0, selected_indices))

        selected_features = torch.stack(selected_feature_batches, dim=0)
        projector = self.get_model().mm_projector
        try:
            p0 = next(projector.parameters())
            selected_features = selected_features.to(device=p0.device, dtype=p0.dtype, non_blocking=True)
        except StopIteration:
            pass
        return projector(selected_features)

    model.encode_images_with_anchor_expansion_pruning = MethodType(_encode_images_with_anchor_expansion_pruning, model)

    if not hasattr(model, "_llava_encode_images_orig"):
        model._llava_encode_images_orig = model.encode_images

    def _encode_images_patched(self, images, image_sizes=None):
        if bool(getattr(self.config, "llava_prune_enable", False)):
            if (not isinstance(images, list)) and isinstance(images, torch.Tensor) and images.ndim == 4:
                return self.encode_images_with_anchor_expansion_pruning(images, image_sizes=image_sizes)
        try:
            return self._llava_encode_images_orig(images, image_sizes=image_sizes)
        except TypeError as e:
            if "image_sizes" in str(e):
                return self._llava_encode_images_orig(images)
            raise

    model.encode_images = MethodType(_encode_images_patched, model)
    model._llava_prune_runtime_patched = True
