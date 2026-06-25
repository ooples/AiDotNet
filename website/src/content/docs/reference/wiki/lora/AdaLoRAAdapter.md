---
title: "AdaLoRAAdapter<T>"
description: "Adaptive Low-Rank Adaptation (AdaLoRA) adapter that dynamically allocates parameter budgets among weight matrices."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LoRA.Adapters`

Adaptive Low-Rank Adaptation (AdaLoRA) adapter that dynamically allocates parameter budgets among weight matrices.

## For Beginners

AdaLoRA is like smart LoRA that learns which parts of the adaptation matter most.

Think of standard LoRA as giving every layer the same budget (rank=8 everywhere).
AdaLoRA is smarter:

- Some layers get more budget (rank=16) because they're important for the task
- Other layers get less budget (rank=2) because small changes are enough
- The model learns this automatically during training

How it works:

1. Start with a large rank (e.g., maxRank=32)
2. During training, track how important each component is
3. Prune components with low importance scores
4. Focus parameters on what actually helps

Benefits:

- More parameter-efficient than fixed-rank LoRA
- Better performance with same parameter budget
- Automatically finds optimal rank per layer

Reference: "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning" (ICLR 2023)
https://arxiv.org/abs/2303.10512

## How It Works

AdaLoRA improves upon standard LoRA by dynamically adjusting the rank allocation based on importance scores.
Instead of using a fixed rank for all weight matrices, AdaLoRA:

- Starts with a maximum rank and adaptively reduces it during training
- Computes importance scores for each singular value component
- Prunes less important components to focus parameter budget on critical adaptations
- Allows different layers to have different effective ranks

This leads to more efficient parameter usage compared to fixed-rank LoRA, especially for large models
where some layers need more adaptation capacity than others.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdaLoRAAdapter(ILayer<>,Int32,Double,Boolean,Double,Int32,Int32,Double)` | Initializes a new AdaLoRA adapter with adaptive rank allocation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentRank` | Gets the current active rank after pruning. |
| `MaxRank` | Gets the maximum rank this adapter can use. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExpandRank(Int32)` | Expands the rank by adding new components (for cases where more capacity is needed). |
| `Forward(Tensor<>)` | Performs the forward pass using only the top-k most important singular values. |
| `GetImportanceScores` | Gets a copy of the current importance scores. |
| `MergeToOriginalLayer` | Merges the AdaLoRA adaptation into the base layer and returns the merged layer. |
| `PruneRank` | Prunes low-importance singular value components to reduce rank. |
| `SyncMatricesToParameters(Matrix<>,Matrix<>)` | Packs the current matrix A and matrix B values into the LoRA layer's flat parameter vector while preserving any tail parameters (biases, scale factors) that exist beyond the two matrices. |
| `UpdateImportanceScores` | Updates importance scores based on current gradient magnitudes. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_activeIndices` | Physical rank indices in `matrixA`/`matrixB` that remain active after pruning. |
| `_currentRank` | Current active rank after pruning. |
| `_importanceScoreEMA` | Exponential moving average factor for importance score updates. |
| `_importanceScores` | Importance scores for each singular value component. |
| `_maxRank` | Maximum possible rank for this adapter. |
| `_minRank` | Minimum rank to maintain (prevents pruning below this threshold). |
| `_pruningInterval` | Number of training steps between rank pruning operations. |
| `_rankPruningThreshold` | Pruning fraction (0.0–1.0): the bottom `_rankPruningThreshold` fraction of active components, ranked by importance score, is dropped on each prune cycle. |
| `_rng` | Static random number generator for thread-safe initialization. |
| `_stepCount` | Current training step counter. |

