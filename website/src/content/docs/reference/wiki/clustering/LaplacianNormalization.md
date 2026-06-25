---
title: "LaplacianNormalization"
description: "Laplacian matrix normalization types."
section: "API Reference"
---

`Enums` · `AiDotNet.Clustering.Options`

Laplacian matrix normalization types.

## Fields

| Field | Summary |
|:-----|:--------|
| `Normalized` | Symmetric normalized: L = D^(-1/2) * (D-W) * D^(-1/2). |
| `RandomWalk` | Random walk normalization: L = D^(-1) * (D-W). |
| `Unnormalized` | Unnormalized Laplacian: L = D - W. |

