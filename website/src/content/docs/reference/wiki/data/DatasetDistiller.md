---
title: "DatasetDistiller"
description: "Performs dataset distillation to synthesize a compact representative dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Quality`

Performs dataset distillation to synthesize a compact representative dataset.

## How It Works

Dataset distillation creates a small synthetic dataset that captures the essential
patterns of the original training data. Uses a gradient-based optimization approach
to iteratively refine synthetic samples toward class centroids.

## Methods

| Method | Summary |
|:-----|:--------|
| `Distill(Double[][],Int32[])` | Distills a dataset by computing class-wise centroids and refining them. |

