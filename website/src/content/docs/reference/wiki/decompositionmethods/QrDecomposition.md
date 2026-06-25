---
title: "QrDecomposition<T>"
description: "Performs QR decomposition on a matrix, factoring it into an orthogonal matrix Q and an upper triangular matrix R."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Performs QR decomposition on a matrix, factoring it into an orthogonal matrix Q and an upper triangular matrix R.

## For Beginners

QR decomposition breaks down a matrix into two parts - Q (which has perpendicular columns with length 1)
and R (which is triangular with zeros below the diagonal). This is useful for solving equations and other matrix operations.
Think of it like factoring a number into its prime components, but for matrices.

## How It Works

QR decomposition factors a matrix A into the product of an orthogonal matrix Q and an upper triangular
matrix R. This decomposition is widely used in solving linear systems, computing eigenvalues, and least
squares problems. Multiple algorithms are available, each with different performance characteristics.

Real-world applications:

- Solving systems of linear equations
- Computing eigenvalues and eigenvectors
- Least squares regression in statistics

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QrDecomposition(Matrix<>,QrAlgorithmType)` | Creates a new QR decomposition of the specified matrix. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Q` | Gets the orthogonal matrix Q from the decomposition. |
| `R` | Gets the upper triangular matrix R from the decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BackSubstitution(Matrix<>,Vector<>)` | Solves a system of linear equations using back substitution on an upper triangular matrix. |
| `ComputeDecomposition(Matrix<>,QrAlgorithmType)` | Selects and applies the appropriate QR decomposition algorithm. |
| `ComputeQrGivens(Matrix<>)` | Computes QR decomposition using Givens rotations. |
| `ComputeQrGramSchmidt(Matrix<>)` | Computes QR decomposition using the classical Gram-Schmidt process. |
| `ComputeQrHouseholder(Matrix<>)` | Computes QR decomposition using Householder reflections. |
| `ComputeQrIterativeGramSchmidt(Matrix<>)` | Computes QR decomposition using the Iterative Gram-Schmidt process for enhanced numerical stability. |
| `ComputeQrModifiedGramSchmidt(Matrix<>)` | Computes QR decomposition using the Modified Gram-Schmidt process, which is more numerically stable. |
| `Decompose` | Performs the QR decomposition. |
| `Solve(Vector<>)` | Solves the linear system Ax = b using the QR decomposition. |

