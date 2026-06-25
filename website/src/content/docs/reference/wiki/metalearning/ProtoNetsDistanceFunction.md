---
title: "ProtoNetsDistanceFunction"
description: "Distance functions supported by Prototypical Networks for measuring similarity between embeddings."
section: "API Reference"
---

`Enums` · `AiDotNet.MetaLearning.Options`

Distance functions supported by Prototypical Networks for measuring similarity between embeddings.

## For Beginners

The distance function determines how we measure
"closeness" between examples. Euclidean is like measuring with a ruler,
Cosine measures the angle between vectors, and Mahalanobis accounts for
correlations in the data.

## How It Works

The choice of distance function affects how the model measures similarity between
query embeddings and class prototypes. Different distance functions have different
properties and may work better for different types of data.

## Fields

| Field | Summary |
|:-----|:--------|
| `Cosine` | Cosine distance (1 - cosine similarity). |
| `Euclidean` | Standard Euclidean (L2) distance. |
| `Mahalanobis` | Mahalanobis distance with learned or estimated covariance. |

