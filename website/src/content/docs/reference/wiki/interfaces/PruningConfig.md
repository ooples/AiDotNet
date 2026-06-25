---
title: "PruningConfig"
description: "Configuration for pruning operations."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Interfaces`

Configuration for pruning operations.

## How It Works

**Layer-aware pruning:** When used with models implementing `ILayeredModel`,
the `CategorySparsityTargets` property enables per-category sparsity levels.
For example, attention layers can be pruned less aggressively than dense layers.

## Properties

| Property | Summary |
|:-----|:--------|
| `CategorySparsityTargets` | Per-category sparsity targets (category → sparsity). |
| `FineTuneAfterPruning` | Whether to fine-tune after pruning. |
| `FineTuningEpochs` | Number of fine-tuning epochs after pruning. |
| `GradualPruning` | Whether to use gradual pruning (multiple iterations). |
| `InitialSparsity` | Initial sparsity for gradual pruning. |
| `LayerSparsityTargets` | Per-layer sparsity targets (layer name → sparsity). |
| `LayerWiseSparsity` | Whether to apply different sparsity per layer (sensitivity-based). |
| `OutputFormat` | Output sparse format for storage. |
| `Pattern` | Sparsity pattern to use. |
| `PruningIterations` | Number of pruning iterations for gradual pruning. |
| `SparsityM` | M value for N:M sparsity (group size). |
| `SparsityN` | N value for N:M sparsity (zeros per group). |
| `TargetSparsity` | Target sparsity level (0.0 to 1.0). |

