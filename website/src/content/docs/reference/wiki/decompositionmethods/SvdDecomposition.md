---
title: "SvdDecomposition<T>"
description: "Implements Singular Value Decomposition (SVD) for matrices."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Implements Singular Value Decomposition (SVD) for matrices.

## For Beginners

SVD is a way to break down a matrix into three simpler matrices (U, S, and V^T).
Think of it like factoring a number, but for matrices. This decomposition is useful for data
compression, noise reduction, and solving systems of equations.

## How It Works

Singular Value Decomposition (SVD) factors a matrix A into the product A = U * S * V^T, where U and V
are orthogonal matrices and S is a diagonal matrix containing singular values. SVD is one of the most
important matrix decompositions with applications across many fields of science and engineering.

Real-world applications:

- Data compression and dimensionality reduction (PCA)
- Image compression and denoising
- Recommender systems (collaborative filtering)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SvdDecomposition(Matrix<>,SvdAlgorithmType)` | Initializes a new instance of the SVD decomposition for the specified matrix. |

## Properties

| Property | Summary |
|:-----|:--------|
| `S` | Gets the singular values vector. |
| `U` | Gets the left singular vectors matrix. |
| `Vt` | Gets the transposed right singular vectors matrix. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyHouseholderLeft(Matrix<>,Vector<>,Int32,Int32,Int32,Int32)` | Applies a Householder transformation from the left side to a matrix. |
| `ApplyHouseholderRight(Matrix<>,Vector<>,Int32,Int32,Int32,Int32)` | Applies a Householder transformation from the right side to a matrix. |
| `Bidiagonalize(Matrix<>,Matrix<>,Matrix<>)` | Performs bidiagonalization of a matrix as part of the SVD process. |
| `ComputeDecomposition(Matrix<>,SvdAlgorithmType)` | Selects and applies the appropriate SVD algorithm based on the specified type. |
| `ComputeDiagonalElements(Matrix<>)` | Extracts the diagonal elements from a matrix as singular values. |
| `ComputeSvdDefault(Matrix<>)` | Computes the Singular Value Decomposition using the default algorithm. |
| `ComputeSvdDividedAndConquer(Matrix<>)` | Computes the Singular Value Decomposition using a divide-and-conquer approach. |
| `ComputeSvdGolubReinsch(Matrix<>)` | Computes the SVD using the Golub-Reinsch algorithm. |
| `ComputeSvdJacobi(Matrix<>)` | Computes the Singular Value Decomposition using the Jacobi method. |
| `ComputeSvdPowerIteration(Matrix<>)` | Computes the Singular Value Decomposition using the power iteration method. |
| `ComputeSvdRandomized(Matrix<>)` | Computes the Singular Value Decomposition using a randomized algorithm. |
| `ComputeTruncatedSvd(Matrix<>)` | Computes a truncated Singular Value Decomposition that keeps only the most significant singular values. |
| `Decompose` | Performs the SVD decomposition. |
| `DiagonalizeGolubReinsch(Matrix<>,Matrix<>,Matrix<>)` | Diagonalizes a bidiagonal matrix using the Golub-Reinsch algorithm. |
| `GenerateRandomMatrix(Int32,Int32)` | Generates a random matrix with values between -1 and 1. |
| `GolubKahanStep(Vector<>,Vector<>,Int32,Int32,Matrix<>,Matrix<>)` | Performs a Golub-Kahan SVD step on a bidiagonal matrix. |
| `Solve(Vector<>)` | Solves a linear system Ax = b using the SVD decomposition. |
| `SortSingularValues(Vector<>,Matrix<>,Matrix<>)` | Sorts singular values in descending order and rearranges the corresponding columns/rows in U and VT. |

