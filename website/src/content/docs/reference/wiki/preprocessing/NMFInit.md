---
title: "NMFInit"
description: "Specifies the initialization method for NMF."
section: "API Reference"
---

`Enums` · `AiDotNet.Preprocessing.DimensionalityReduction`

Specifies the initialization method for NMF.

## Fields

| Field | Summary |
|:-----|:--------|
| `NNDSVD` | Non-negative Double SVD initialization (faster convergence). |
| `NNDSVDa` | NNDSVD with zeros filled with small values (better for sparse data). |
| `Random` | Random non-negative initialization. |

