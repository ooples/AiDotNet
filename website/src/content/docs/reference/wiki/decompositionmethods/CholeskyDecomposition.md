---
title: "CholeskyDecomposition<T>"
description: "Implements the Cholesky decomposition for symmetric positive definite matrices."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Implements the Cholesky decomposition for symmetric positive definite matrices.

## For Beginners

Cholesky decomposition is like taking the square root of a matrix.
Just as 25 = 5 × 5, this method breaks down special matrices into L × L^T, where L is
a simpler triangular matrix. This makes complex calculations much faster and more accurate.

## How It Works

The Cholesky decomposition breaks down a symmetric positive definite matrix into
the product of a lower triangular matrix and its transpose (A = L * L^T).
This is useful for solving linear systems and matrix inversion more efficiently.

Real-world applications:

- Solving systems of linear equations in scientific computing
- Covariance matrix decomposition in statistics and machine learning
- Efficient simulation of correlated random variables

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CholeskyDecomposition(Matrix<>,CholeskyAlgorithmType)` | Initializes a new instance of the Cholesky decomposition for the specified matrix. |

## Properties

| Property | Summary |
|:-----|:--------|
| `L` | Gets the lower triangular matrix L from the decomposition A = L * L^T. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BackSubstitution(Matrix<>,Vector<>)` | Performs back substitution to solve L^T*x = y. |
| `ComputeBlockCholesky(Matrix<>)` | Computes the Cholesky decomposition using a block algorithm for large matrices. |
| `ComputeCholeskyBanachiewicz(Matrix<>)` | Computes the Cholesky decomposition using the Banachiewicz algorithm. |
| `ComputeCholeskyCrout(Matrix<>)` | Computes the Cholesky decomposition using the Crout algorithm. |
| `ComputeCholeskyDefault(Matrix<>)` | Computes the Cholesky decomposition using a default approach. |
| `ComputeCholeskyLDL(Matrix<>)` | Computes the LDL' decomposition and converts it to Cholesky form. |
| `ComputeDecomposition(Matrix<>,CholeskyAlgorithmType)` | Selects and applies the appropriate Cholesky decomposition algorithm. |
| `Decompose` | Performs the Cholesky decomposition. |
| `ForwardSubstitution(Matrix<>,Vector<>)` | Performs forward substitution to solve L*y = b. |
| `Solve(Vector<>)` | Solves the linear system Ax = b using the Cholesky decomposition. |
| `SolveMatrix(Matrix<>)` | Solves a system of linear equations for multiple right-hand sides. |
| `ValidateSymmetric(Matrix<>)` | Validates that the matrix is symmetric. |

