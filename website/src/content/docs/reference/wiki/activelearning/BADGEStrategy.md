---
title: "BADGEStrategy<T, TInput, TOutput>"
description: "BADGE (Batch Active learning by Diverse Gradient Embeddings) strategy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.Strategies.Hybrid`

BADGE (Batch Active learning by Diverse Gradient Embeddings) strategy.

## For Beginners

BADGE is a state-of-the-art strategy that combines uncertainty
and diversity. It uses gradient embeddings (gradients of the loss with respect to
the final layer) to represent samples, then uses k-means++ to select diverse samples.

## How It Works

**How BADGE Works:**

**Key Insight:** Gradient embeddings naturally combine model uncertainty
(large gradients for uncertain samples) with feature diversity (different gradient
directions for different samples).

**When to Use:**

**Reference:** Ash et al. "Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds" (ICLR 2020)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BADGEStrategy(Int32)` | Initializes a new BADGE strategy. |
| `BADGEStrategy(Int32,ActiveLearnerConfig<>)` | Initializes a new BADGE strategy with configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeScores(IFullModel<,,>,IDataset<,,>)` |  |
| `Reset` |  |
| `SelectSamples(IFullModel<,,>,IDataset<,,>,Int32)` |  |
| `UpdateState(Int32[],[])` |  |

