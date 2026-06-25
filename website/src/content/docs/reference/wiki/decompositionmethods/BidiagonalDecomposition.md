---
title: "BidiagonalDecomposition<T>"
description: "Implements the Bidiagonal Decomposition of a matrix, which factors a matrix into U*B*V^T, where U and V are orthogonal matrices and B is a bidiagonal matrix."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Implements the Bidiagonal Decomposition of a matrix, which factors a matrix into U*B*V^T,
where U and V are orthogonal matrices and B is a bidiagonal matrix.

## For Beginners

Bidiagonal decomposition is like organizing a complex matrix into a simpler form
that's easier to work with. Think of it like arranging books on a shelf: instead of having them scattered,
you organize them so most spaces are empty (zeros) and important information (non-zero values) is concentrated
in just two lines. This makes many calculations much faster and more efficient, especially when computing
singular values or solving systems of equations.

## How It Works

Bidiagonal decomposition transforms a matrix A into the form A = U*B*V^T, where:

- U is an orthogonal matrix (left singular vectors)
- B is a bidiagonal matrix (non-zero elements only on main diagonal and superdiagonal)
- V^T is the transpose of an orthogonal matrix (right singular vectors)

Common applications include:

- Computing Singular Value Decomposition (SVD)
- Solving least squares problems
- Computing eigenvalues of symmetric matrices
- Data compression and dimensionality reduction

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BidiagonalDecomposition(Matrix<>,BidiagonalAlgorithmType)` | Creates a new bidiagonal decomposition of the specified matrix. |

## Properties

| Property | Summary |
|:-----|:--------|
| `B` | Gets the bidiagonal matrix in the decomposition. |
| `U` | Gets the left orthogonal matrix in the decomposition. |
| `V` | Gets the right orthogonal matrix in the decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDecomposition(BidiagonalAlgorithmType)` | Performs the actual decomposition computation using the specified algorithm. |
| `Decompose` | Performs the bidiagonal decomposition. |
| `Decompose(BidiagonalAlgorithmType)` | Performs the bidiagonal decomposition using the specified algorithm. |
| `DecomposeGivens` | Performs bidiagonal decomposition using Givens rotations. |
| `DecomposeHouseholder` | Performs bidiagonal decomposition using Householder reflections. |
| `DecomposeLanczos` | Performs bidiagonal decomposition using the Lanczos algorithm. |
| `GetFactors` | Returns the three factor matrices of the bidiagonal decomposition. |
| `GivensRotation(Matrix<>,Matrix<>,Int32,Int32,Int32,Int32,Boolean)` | Applies a Givens rotation to the specified matrices. |
| `HouseholderVector(Vector<>)` | Computes a Householder reflection vector for the given input vector. |
| `Invert` | Computes the inverse of the original matrix A using the bidiagonal decomposition. |
| `Solve(Vector<>)` | Solves the linear system Ax = b using the bidiagonal decomposition. |
| `SolveBidiagonal(Vector<>)` | Solves a bidiagonal system of linear equations. |

