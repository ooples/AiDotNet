---
title: "EigenDecomposition<T>"
description: "Performs eigenvalue decomposition of a matrix, breaking it down into its eigenvalues and eigenvectors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Performs eigenvalue decomposition of a matrix, breaking it down into its eigenvalues and eigenvectors.

## For Beginners

Eigenvalue decomposition finds special directions (eigenvectors) in which a matrix
acts like simple scaling. When you multiply the matrix by an eigenvector, you get the same vector back
but scaled by a number (the eigenvalue). Think of it like finding the "natural directions" of a transformation.

## How It Works

Eigenvalue decomposition is a way to factorize a square matrix into a set of eigenvectors and eigenvalues.
This is useful in many applications including principal component analysis, vibration analysis,
and solving systems of differential equations.

Real-world applications:

- Principal Component Analysis (PCA) for data dimensionality reduction
- Vibration analysis in mechanical engineering
- Google's PageRank algorithm

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EigenDecomposition(Matrix<>,EigenAlgorithmType)` | Creates a new eigenvalue decomposition for the specified matrix. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EigenValues` | Gets the eigenvalues of the decomposed matrix. |
| `EigenVectors` | Gets the eigenvectors of the decomposed matrix. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDecomposition(Matrix<>,EigenAlgorithmType)` | Selects and applies the appropriate eigenvalue decomposition algorithm. |
| `ComputeEigenJacobi(Matrix<>)` | Computes eigenvalues and eigenvectors using the Jacobi method. |
| `ComputeEigenPowerIteration(Matrix<>)` | Computes eigenvalues and eigenvectors using the Power Iteration method. |
| `ComputeEigenQR(Matrix<>)` | Computes eigenvalues and eigenvectors using the QR algorithm with Wilkinson shift. |
| `Decompose` | Performs the eigenvalue decomposition. |
| `GetMachineEpsForType` | Type-appropriate machine epsilon. |
| `Invert` | Computes the inverse of the original matrix using the eigenvalue decomposition. |
| `Solve(Vector<>)` | Solves a system of linear equations Ax = b using the eigenvalue decomposition. |
| `SumAbsUpperOffDiagonal(Matrix<>)` | Sum of `\|a[i,j]\|` over the strict upper triangle (i < j). |
| `SumSquaredEntries(Matrix<>)` | Sum of squared entries — used for the Frobenius-norm-based convergence threshold in the sweep-based Jacobi loop. |
| `SumSquaredOffDiagonal(Matrix<>)` | Sum of squared off-diagonal entries (i ≠ j). |

