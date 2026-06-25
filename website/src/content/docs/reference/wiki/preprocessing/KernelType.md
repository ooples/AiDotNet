---
title: "KernelType"
description: "Specifies the kernel type for Kernel PCA."
section: "API Reference"
---

`Enums` · `AiDotNet.Preprocessing.DimensionalityReduction`

Specifies the kernel type for Kernel PCA.

## Fields

| Field | Summary |
|:-----|:--------|
| `Linear` | Linear kernel: K(x, y) = x · y |
| `Polynomial` | Polynomial kernel: K(x, y) = (γ(x · y) + c₀)^d |
| `RBF` | Radial Basis Function (Gaussian): K(x, y) = exp(-γ\|\|x-y\|\|²) |
| `Sigmoid` | Sigmoid (hyperbolic tangent): K(x, y) = tanh(γ(x · y) + c₀) |

