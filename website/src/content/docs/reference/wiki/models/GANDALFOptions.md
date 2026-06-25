---
title: "GANDALFOptions<T>"
description: "Configuration options for GANDALF (Gated Additive Neural Decision Forest)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for GANDALF (Gated Additive Neural Decision Forest).

## For Beginners

GANDALF is like having a smart feature selector combined
with a forest of decision trees.

Key ideas:

1. **Gated Feature Selection**: Learns which features matter for each prediction
2. **Soft Decision Trees**: Trees with smooth (differentiable) decisions
3. **Additive Ensemble**: Trees are combined by adding their predictions

Why this works well:

- Automatic feature importance learning (no manual selection needed)
- Interpretable structure (can see which features and paths are used)
- Combines benefits of neural networks and decision trees
- Good handling of both numerical and categorical features

Example:

## How It Works

GANDALF combines gated feature selection with neural decision trees in an additive
ensemble. It learns which features are important through attention-based gating
and makes predictions through soft decision trees.

Reference: "GANDALF: Gated Adaptive Network for Deep Automated Learning of Features" (2022)

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `GatingHiddenDimension` | Gets or sets the hidden dimension for gating network. |
| `InitScale` | Gets or sets the initialization scale for tree parameters. |
| `LeafDimension` | Gets or sets the leaf dimension (output dimension per leaf). |
| `NumGatingLayers` | Gets or sets the number of gating layers. |
| `NumInternalNodes` | Gets the number of internal nodes per tree (2^depth - 1). |
| `NumLeaves` | Gets the number of leaves per tree (2^depth). |
| `NumTrees` | Gets or sets the number of trees in the ensemble. |
| `Temperature` | Gets or sets the temperature for soft tree decisions. |
| `TreeDepth` | Gets or sets the depth of each tree. |
| `UseBatchNorm` | Gets or sets whether to use batch normalization. |
| `UseFeatureGating` | Gets or sets whether to use feature-specific gating. |
| `UseResidualGating` | Gets or sets whether to use residual connections in gating. |
| `WeightDecay` | Gets or sets the weight decay for regularization. |

