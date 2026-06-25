---
title: "EigenSolver"
description: "Eigenvalue solver methods."
section: "API Reference"
---

`Enums` · `AiDotNet.Clustering.Options`

Eigenvalue solver methods.

## Fields

| Field | Summary |
|:-----|:--------|
| `Amg` | Use AMG for algebraic multigrid preconditioned LOBPCG. |
| `Arpack` | Use ARPACK for sparse matrices (default for large datasets). |
| `Full` | Full eigenvalue decomposition (only for small datasets). |
| `Lobpcg` | Use LOBPCG for sparse matrices with preconditioner. |

