---
title: "LdlDecomposition<T>"
description: "Performs LDL decomposition on a symmetric matrix, factoring it into a lower triangular matrix L and a diagonal matrix D such that A = LDL^T."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Performs LDL decomposition on a symmetric matrix, factoring it into a lower triangular matrix L
and a diagonal matrix D such that A = LDL^T.

## For Beginners

LDL decomposition breaks down a symmetric matrix into simpler parts:
L (a lower triangular matrix with values only on and below the diagonal) and D (a diagonal matrix
with values only on the diagonal). This decomposition is useful for solving linear systems,
calculating determinants, and inverting matrices more efficiently than working with the original matrix.

## How It Works

LDL decomposition factors a symmetric matrix A into the product A = LDL^T, where L is a lower
triangular matrix with ones on the diagonal, and D is a diagonal matrix. This decomposition is
particularly useful for symmetric matrices and avoids computing square roots, making it more
numerically stable than Cholesky decomposition in some cases.

Real-world applications:

- Solving systems of linear equations in optimization
- Covariance matrix analysis in statistics
- Kalman filtering in signal processing

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LdlDecomposition(Matrix<>,LdlAlgorithmType)` | Initializes a new instance of the LDL decomposition for the specified matrix. |

## Properties

| Property | Summary |
|:-----|:--------|
| `D` | The diagonal matrix D from the decomposition, stored as a vector of the diagonal elements. |
| `L` | The lower triangular matrix L from the decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDecomposition(LdlAlgorithmType)` | Performs the actual decomposition computation using the specified algorithm. |
| `Decompose` | Performs the LDL decomposition. |
| `DecomposeCholesky` | Performs LDL decomposition using the Cholesky-based algorithm. |
| `DecomposeCrout` | Performs LDL decomposition using the Crout algorithm. |
| `GetFactors` | Returns the L and D factors from the decomposition. |
| `Invert` | Calculates the inverse of the original matrix using the LDL decomposition. |
| `Solve(Vector<>)` | Solves the linear system Ax = b using the LDL decomposition. |

