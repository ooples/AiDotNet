---
title: "LqDecomposition<T>"
description: "Performs LQ decomposition on a matrix, factoring it into a lower triangular matrix L and an orthogonal matrix Q."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Performs LQ decomposition on a matrix, factoring it into a lower triangular matrix L and an orthogonal matrix Q.

## For Beginners

LQ decomposition breaks down a matrix A into two components:
L (a lower triangular matrix with values only on and below the diagonal) and
Q (an orthogonal matrix with columns that are perpendicular to each other).
This decomposition is useful for solving linear systems, least squares problems,
and other numerical linear algebra tasks.

## How It Works

LQ decomposition factors a matrix A into the product A = LQ, where L is a lower triangular matrix
and Q is an orthogonal matrix. This decomposition is the transpose version of QR decomposition and
is particularly useful when working with matrices that have more columns than rows.

Real-world applications:

- Solving underdetermined systems of equations
- Least squares problems with wide matrices
- Numerical stability in various computations

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LqDecomposition(Matrix<>,LqAlgorithmType)` | Initializes a new instance of the LqDecomposition class and performs the decomposition. |

## Properties

| Property | Summary |
|:-----|:--------|
| `L` | Gets the lower triangular matrix L from the decomposition. |
| `Q` | Gets the orthogonal matrix Q from the decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDecomposition(Matrix<>,LqAlgorithmType)` | Selects and applies the appropriate decomposition algorithm. |
| `ComputeLqGivens(Matrix<>)` | Computes the LQ decomposition using Givens rotations. |
| `ComputeLqGramSchmidt(Matrix<>)` | Computes the LQ decomposition using the Gram-Schmidt orthogonalization process. |
| `ComputeLqHouseholder(Matrix<>)` | Computes the LQ decomposition using the Householder reflection method. |
| `ComputeQrGivens(Matrix<>)` | Computes QR decomposition using Givens rotations (internal helper for LQ). |
| `ComputeQrGramSchmidt(Matrix<>)` | Computes QR decomposition using Gram-Schmidt (internal helper for LQ). |
| `ComputeQrHouseholder(Matrix<>)` | Computes QR decomposition using Householder reflections (internal helper for LQ). |
| `Decompose` | Performs the LQ decomposition. |
| `ForwardSubstitution(Matrix<>,Vector<>)` | Solves a linear system Lx = b where L is a lower triangular matrix. |
| `Solve(Vector<>)` | Solves the linear system Ax = b using the LQ decomposition. |

