---
title: "NMINormalization"
description: "Normalization methods for NMI."
section: "API Reference"
---

`Enums` · `AiDotNet.Clustering.Evaluation`

Normalization methods for NMI.

## Fields

| Field | Summary |
|:-----|:--------|
| `Arithmetic` | Arithmetic mean of entropies: (H(U) + H(V)) / 2 |
| `Geometric` | Geometric mean of entropies: sqrt(H(U) * H(V)) |
| `Max` | Maximum of entropies: max(H(U), H(V)) |
| `Min` | Minimum of entropies: min(H(U), H(V)) |

