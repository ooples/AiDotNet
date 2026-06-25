---
title: "TridiagonalDecomposition<T>"
description: "Represents a tridiagonal decomposition of a matrix, which decomposes a matrix A into Q*T*Q^T, where Q is orthogonal and T is tridiagonal."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Represents a tridiagonal decomposition of a matrix, which decomposes a matrix A into Q*T*Q^T,
where Q is orthogonal and T is tridiagonal.

## For Beginners

A tridiagonal matrix is a special type of matrix where non-zero elements
can only appear on the main diagonal and the diagonals directly above and below it.
This decomposition transforms a complex matrix into this simpler form, making many
calculations much faster and more efficient.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TridiagonalDecomposition(Matrix<>,TridiagonalAlgorithmType)` | Initializes a new instance of the TridiagonalDecomposition class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `QMatrix` | Gets the orthogonal matrix Q in the decomposition A = Q*T*Q^T. |
| `TMatrix` | Gets the tridiagonal matrix T in the decomposition A = Q*T*Q^T. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDecomposition(TridiagonalAlgorithmType)` | Computes the tridiagonal decomposition using the specified algorithm. |
| `Decompose` | Performs the tridiagonal decomposition. |
| `DecomposeGivens` | Performs tridiagonal decomposition using the Givens algorithm. |
| `DecomposeHouseholder` | Performs tridiagonal decomposition using the Householder algorithm. |
| `DecomposeLanczos` | Performs tridiagonal decomposition using the Lanczos algorithm. |
| `GetFactors` | Returns the matrices Q and T from the decomposition A = Q*T*Q^T. |
| `Invert` | Calculates the inverse of the original matrix A using the tridiagonal decomposition. |
| `InvertTridiagonal` | Calculates the inverse of the tridiagonal matrix T. |
| `Solve(Vector<>)` | Solves the linear system Ax = b using the tridiagonal decomposition. |
| `SolveTridiagonal(Vector<>)` | Solves a tridiagonal linear system of equations Tx = b. |

