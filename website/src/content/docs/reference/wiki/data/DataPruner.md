---
title: "DataPruner"
description: "Prunes (removes) training samples based on difficulty/importance scores."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Quality`

Prunes (removes) training samples based on difficulty/importance scores.

## How It Works

Data pruning removes easy or redundant samples to reduce training set size
while preserving model quality. Supports multiple scoring strategies:
confidence-based, forgetting events, EL2N, and GraNd scores.
Requires per-sample training signals collected during a warmup phase.

## Methods

| Method | Summary |
|:-----|:--------|
| `Prune(Double[])` | Identifies samples to prune using the configured strategy. |
| `PruneByConfidence(Double[])` | Identifies samples to prune based on confidence scores. |
| `PruneByEL2N(Double[])` | Identifies samples to prune based on EL2N (Error L2 Norm) scores. |
| `PruneByForgetting(Int32[])` | Identifies samples to prune based on forgetting event counts. |
| `PruneByGraNd(Double[])` | Identifies samples to prune based on GraNd (Gradient Norm) scores. |

