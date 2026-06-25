---
title: "InducingPointKernel<T>"
description: "Inducing Point Kernel for sparse Gaussian Process approximations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Inducing Point Kernel for sparse Gaussian Process approximations.

## For Beginners

The Inducing Point Kernel is fundamental to making GPs scale to large datasets.
Instead of working with all N training points (O(N³) complexity), we use a smaller set
of M "inducing points" (O(NM² + M³) complexity, where M << N).

The key idea: Instead of modeling the full function f, we model:

1. The function values at M inducing points: u = f(Z)
2. The conditional distribution: f(x) | u

The approximation makes f(x) conditionally independent given the inducing values u.
The closer the inducing points are to your data, the better the approximation.

This kernel computes the "Q_ff" approximation:
Q(x, x') = k(x, Z) × k(Z, Z)⁻¹ × k(Z, x')

Where:

- Z are the inducing point locations (M × d)
- k(x, Z) is the cross-covariance (1 × M)
- k(Z, Z) is the inducing point covariance (M × M)

This is the Nyström approximation of the full kernel matrix.

## How It Works

Usage patterns:

- Use with SparseVariationalGaussianProcess for scalable GP regression
- Inducing points can be learned via optimization
- Good initialization: subset of training data or k-means centroids

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InducingPointKernel(IKernelFunction<>,Matrix<>,Double)` | Initializes an Inducing Point Kernel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseKernel` | Gets the base kernel. |
| `NumInducingPoints` | Gets the number of inducing points. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the approximate kernel value (Nyström approximation). |
| `CholeskyDecomposition(Matrix<>)` | Cholesky decomposition. |
| `ComputeCrossCovariance(Matrix<>)` | Computes the cross-covariance between test points and inducing points. |
| `EstimateApproximationError(Matrix<>)` | Estimates the approximation quality at a set of test points. |
| `GetCholeskyFactor` | Gets the Cholesky factor of k(Z, Z). |
| `GetInducingPointCovariance` | Gets the inducing point covariance matrix k(Z, Z). |
| `GetInducingPoints` | Gets a copy of the inducing points. |
| `GetRow(Matrix<>,Int32)` | Extracts a row from a matrix. |
| `PrecomputeMatrices` | Precomputes the inducing point matrices for efficiency. |
| `SolveLowerTriangular(Matrix<>,Vector<>)` | Solves Lx = b for x. |
| `UpdateInducingPoints(Matrix<>)` | Updates the inducing point locations. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_Kzz` | Precomputed k(Z, Z). |
| `_Lzz` | Cholesky decomposition of k(Z, Z). |
| `_baseKernel` | The base kernel function. |
| `_inducingPoints` | The inducing point locations. |
| `_jitter` | Small regularization constant. |
| `_matricesValid` | Whether precomputed matrices are valid. |
| `_numInducingPoints` | Number of inducing points. |
| `_numOps` | Operations for performing numeric calculations with type T. |

