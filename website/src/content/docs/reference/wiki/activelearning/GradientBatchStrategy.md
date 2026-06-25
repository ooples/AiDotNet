---
title: "GradientBatchStrategy<T, TInput, TOutput>"
description: "Gradient-based batch selection strategy using gradient embeddings (BADGE-style)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.Batch`

Gradient-based batch selection strategy using gradient embeddings (BADGE-style).

## For Beginners

BADGE (Batch Active learning by Diverse Gradient Embeddings)
is a state-of-the-art method that combines uncertainty and diversity. It uses gradient
embeddings - vectors derived from the model's gradients - to represent samples.

## How It Works

**How It Works:**

**Why Gradient Embeddings?**

**Reference:** Ash et al. "Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds" (ICLR 2020)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientBatchStrategy` | Initializes a new GradientBatchStrategy with default settings. |
| `GradientBatchStrategy(Boolean)` | Initializes a new GradientBatchStrategy with specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DiversityTradeoff` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDiversity(,)` |  |
| `ComputeGradientEmbeddings(IFullModel<,,>,IDataset<,,>)` |  |
| `KMeansPlusPlusSelection(Matrix<>,Int32)` |  |
| `SelectBatch(Int32[],Vector<>,IDataset<,,>,Int32)` |  |

