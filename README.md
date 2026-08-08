# AnchorPrune

**AnchorPrune: Relevance-Anchored Contextual Expansion for Visual Token Pruning**  
Kyuan Oh and Bumsoo Kim  
Chung-Ang University

<p>
  <img src="https://img.shields.io/badge/ECCV-2026-4c8eda" alt="ECCV 2026">
  <a href="https://arxiv.org/abs/2607.07033"><img src="https://img.shields.io/badge/arXiv-2607.07033-b31b1b" alt="arXiv 2607.07033"></a>
  <img src="https://img.shields.io/badge/License-Apache--2.0-181717" alt="Apache 2.0">
</p>

Official implementation of the [ECCV 2026 paper](https://arxiv.org/abs/2607.07033).

AnchorPrune is a training-free visual-token pruning method for efficient vision-language model inference. The method first constructs a protected relevance anchor from query-conditioned visual-token priorities, and then expands the retained set with importance-weighted contextual novelty. This relevance-anchored ordering prevents indispensable query evidence from being displaced by later diversity or coverage optimization.

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Supported Backbones](#supported-backbones)
- [Repository Layout](#repository-layout)
- [Reproduction](#reproduction)
- [Paper Settings](#paper-settings)
- [Benchmark Task Names](#benchmark-task-names)
- [Core Selector Usage](#core-selector-usage)
- [Citation](#citation)
- [License and Acknowledgements](#license-and-acknowledgements)

## Overview

AnchorPrune is designed as a lightweight inference-time module:

- **Training-free:** no finetuning, model-parameter updates, or architecture retraining.
- **Relevance-anchored:** query-critical visual evidence is protected before contextual expansion.
- **Context-aware:** the remaining budget is allocated by maximizing `p_i · Delta(i; S)`, selecting tokens that are both globally informative and non-redundant.
- **Architecture-aware:** verified integrations are provided for LLaVA and Qwen.

## Installation

Clone the repository:

```bash
git clone https://github.com/MULTI-cau/AnchorPrune.git
cd AnchorPrune
```

LLaVA and Qwen are evaluated with separate dependency stacks. We recommend creating one environment per backbone.

### LLaVA

```bash
conda create -n anchorprune-llava python=3.10 -y
conda activate anchorprune-llava
pip install -e ".[llava]"
```

### Qwen

```bash
conda create -n anchorprune-qwen python=3.10 -y
conda activate anchorprune-qwen
pip install -e ".[qwen]"
```

## Supported Backbones

| Backbone | Unpruned visual tokens | Paper budgets |
| --- | ---: | ---: |
| LLaVA-1.5-7B | 576 | 32, 64, 128 |
| Qwen2.5-VL-7B | 1,296 at `1008 x 1008` | 64, 128, 256 |

The current release focuses on image-model evaluation. New backbones can be integrated by providing the token-level relevance, feature, and importance signals consumed by the model-agnostic selector in `anchorprune/selection.py`.

## Repository Layout

```text
AnchorPrune/
|-- anchorprune/             # Core selector and model-specific adapters
|   |-- selection.py         # Model-agnostic relevance-anchored expansion
|   |-- llava.py             # LLaVA pruning integration
|   |-- qwen.py              # Qwen pruning integration
|   `-- README.md            # Core module documentation
|-- lmms-eval/               # Lightweight LMMs-Eval fork for benchmark execution
|-- llava/                   # Compact vendored LLaVA inference backend
|-- scripts/                 # Reproduction scripts
|-- LICENSE
|-- THIRD_PARTY_NOTICES.md
|-- pyproject.toml
`-- README.md
```

The LMMs-Eval wrappers intentionally remain thin. They parse evaluation inputs, attach the AnchorPrune runtime hook, and delegate pruning logic to `anchorprune`:

```text
lmms-eval/lmms_eval/models/simple/llava.py
lmms-eval/lmms_eval/models/simple/qwen2_5_vl.py
```

## Reproduction

All scripts follow the same interface:

```text
./scripts/<script>.sh <K> <K_min> <tasks>
```

where `K` is the retained-token budget, `K_min` is the minimum Stage-1 anchor size, and `tasks` is a comma-separated LMMs-Eval task list.

### LLaVA

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/eval_llava_1_5.sh 32 5 mme
```

### Qwen

The Qwen reproduction path fixes the image resolution to `1008 x 1008`, matching the paper setting.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/eval_qwen2_5_vl.sh 64 10 mme
```

### Multiple Benchmarks

The task argument accepts the standard comma-separated LMMs-Eval format. For example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/eval_llava_1_5.sh \
32 5 \
"vqav2_val,textvqa_val,gqa,scienceqa_img,mme,pope,mmbench_en_dev,mmbench_cn_dev,mmvet"
```

## Paper Settings

The scripts expose `K` and `K_min` because these are the primary budget choices. The remaining method choices are fixed to the paper configuration.

| Component | Released setting |
| --- | --- |
| Stage-1 allocation | adaptive protected relevance anchoring |
| Novelty threshold | `tau = 0.2` |
| Patience | cumulative, `P = 3` |
| Stage-1 upper bound | `K_max = 0.5K` |
| Stage-2 objective | maximize `p_i · Delta(i; S)` |
| Output order | native visual-token order |
| LLaVA relevance signal | negated CLIP patch-text similarity |
| Qwen relevance signal | token-wise maximum matching after multimodal projection |

Recommended paper budgets are:

| Backbone | Budget convention | `K` | `K_min` |
| --- | --- | ---: | ---: |
| LLaVA-1.5-7B | total retained tokens | 32 | 5 |
| LLaVA-1.5-7B | total retained tokens | 64 | 10 |
| LLaVA-1.5-7B | total retained tokens | 128 | 20 |
| Qwen2.5-VL-7B | total retained tokens | 64 | 10 |
| Qwen2.5-VL-7B | total retained tokens | 128 | 20 |
| Qwen2.5-VL-7B | total retained tokens | 256 | 40 |

## Benchmark Task Names

| Paper benchmark | LMMs-Eval task |
| --- | --- |
| VQAv2 | `vqav2_val` |
| TextVQA | `textvqa_val` |
| GQA | `gqa` |
| ScienceQA-IMG | `scienceqa_img` |
| MME | `mme` |
| POPE | `pope` |
| MMBench-EN | `mmbench_en_dev` |
| MMBench-CN | `mmbench_cn_dev` |
| MM-Vet | `mmvet` |
| DocVQA | `docvqa_val` |
| AI2D | `ai2d` |
| MMMU | `mmmu_val` |

## Core Selector Usage

The selector can be used independently of LMMs-Eval:

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

See [`anchorprune/README.md`](anchorprune/README.md) for the selector interface and model-specific instantiations.

## Citation

If AnchorPrune is useful for your research, we appreciate citing the paper:

```bibtex
@misc{oh2026anchorprunerelevanceanchoredcontextualexpansion,
  title         = {AnchorPrune: Relevance-Anchored Contextual Expansion for Visual Token Pruning},
  author        = {Kyuan Oh and Bumsoo Kim},
  year          = {2026},
  eprint        = {2607.07033},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2607.07033},
}
```

## License and Acknowledgements

AnchorPrune is released under the Apache License 2.0. See [`LICENSE`](LICENSE) for details.

This release builds on the public implementations and evaluation interfaces of [LLaVA](https://github.com/haotian-liu/LLaVA), [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), and [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). Third-party notices are provided in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
