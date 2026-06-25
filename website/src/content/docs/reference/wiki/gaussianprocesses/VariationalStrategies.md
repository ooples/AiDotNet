---
title: "VariationalStrategies<T>"
description: "Provides variational inference strategies for scalable Gaussian Process inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Provides variational inference strategies for scalable Gaussian Process inference.

## For Beginners

Standard GP inference scales as O(n³), which becomes impractical for
large datasets. Variational inference provides approximations that scale better:

Key ideas:

1. Introduce "inducing points" that summarize the data
2. Approximate the true posterior with a simpler variational distribution
3. Optimize the variational parameters to minimize KL divergence

Common strategies:

- SVGP (Sparse Variational GP): Full variational approximation
- FITC: Fully Independent Training Conditional
- VFE: Variational Free Energy
- KISS-GP: Kernel Interpolation for Scalable Structured GPs

Trade-offs:

- More inducing points = better approximation but slower
- Fewer inducing points = faster but less accurate

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VariationalStrategies` | Initializes a new variational strategies helper. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CholeskyDecomposition(Matrix<>)` | Performs Cholesky decomposition of a symmetric positive definite matrix. |
| `ComputeELBO(Matrix<>,Vector<>,Matrix<>,IKernelFunction<>,Double,Vector<>,Matrix<>)` | Computes the ELBO (Evidence Lower BOund) for sparse GP. |
| `ComputeKernelMatrix(Matrix<>,Matrix<>,IKernelFunction<>)` | Computes a kernel matrix between two sets of points. |
| `ComputeSquaredDistance(Matrix<>,Int32,Matrix<>,Int32)` | Computes squared distance between a data point and a center. |
| `GetRow(Matrix<>,Int32)` | Gets a row from a matrix as a vector. |
| `InitializeVariationalParameters(Matrix<>,Vector<>,Matrix<>,Double)` | Initializes variational parameters for SVGP. |
| `SelectInducingPointsGreedyVariance(Matrix<>,IKernelFunction<>,Int32)` | Selects inducing points using greedy variance reduction. |
| `SelectInducingPointsKMeans(Matrix<>,Int32,Int32,Int32)` | Selects inducing points using k-means clustering on training data. |
| `SolveTriangular(Matrix<>,Matrix<>,Boolean)` | Solves L @ X = B for X where L is lower triangular. |
| `SolveTriangularSystem(Matrix<>,Vector<>)` | Solves L @ L^T @ x = b for x. |

