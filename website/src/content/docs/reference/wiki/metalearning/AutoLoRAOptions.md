---
title: "AutoLoRAOptions<T, TInput, TOutput>"
description: "Configuration options for AutoLoRA (Zhang et al., NAACL 2024)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for AutoLoRA (Zhang et al., NAACL 2024).

## How It Works

AutoLoRA automatically determines optimal per-layer LoRA ranks via meta-learning.
Each rank-1 component has a continuous selection variable α ∈ [0,1] optimized on
validation data (outer loop), while LoRA weights are trained on training data (inner loop).
Final rank is determined by thresholding: k_l = |{α_{l,j} | α_{l,j} ≥ λ}|.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxRank` | Maximum rank per group (initial rank allocation). |
| `NumRankGroups` | Number of rank groups (analogous to layers). |
| `RankRegularization` | Regularization penalty that encourages lower ranks (sparsity). |
| `RankThreshold` | Threshold for rank determination: α_{l,j} ≥ threshold means rank-1 component is kept. |

