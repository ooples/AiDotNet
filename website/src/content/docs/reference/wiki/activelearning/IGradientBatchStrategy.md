---
title: "IGradientBatchStrategy<T, TInput, TOutput>"
description: "Interface for gradient-based batch selection (e.g., BADGE)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ActiveLearning.Interfaces`

Interface for gradient-based batch selection (e.g., BADGE).

## For Beginners

BADGE (Batch Active learning by Diverse Gradient Embeddings)
uses gradient embeddings to represent samples, then selects a diverse set using k-means++.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradientEmbeddings(IFullModel<,,>,IDataset<,,>)` | Computes gradient embeddings for samples. |
| `KMeansPlusPlusSelection(Matrix<>,Int32)` | Selects samples using k-means++ initialization on gradient embeddings. |

