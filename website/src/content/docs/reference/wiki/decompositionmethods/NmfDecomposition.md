---
title: "NmfDecomposition<T>"
description: "Implements Non-negative Matrix Factorization (NMF) for matrices with non-negative elements."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Implements Non-negative Matrix Factorization (NMF) for matrices with non-negative elements.

## For Beginners

NMF is a way to break down a matrix containing only non-negative values
(zero or positive numbers) into two simpler matrices W and H, where V ~= W * H.
Think of it like finding hidden patterns or features in your data.

## How It Works

For example, if you have a matrix of movie ratings (all non-negative), NMF can discover:

- W: How much each user likes different movie genres (features)
- H: How much each movie belongs to different genres (features)

Common applications include:

- Topic modeling in text documents
- Image processing and feature extraction
- Collaborative filtering in recommendation systems
- Audio source separation
- Bioinformatics data analysis

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NmfDecomposition(Matrix<>,Nullable<Int32>,Int32,Double)` | Initializes a new instance of the NMF decomposition for the specified matrix. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Components` | Gets the number of components (features) used in the factorization. |
| `H` | Gets the coefficient matrix H (weights/encodings). |
| `W` | Gets the basis matrix W (features/components). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeNmf(Matrix<>,Int32,Int32,Double)` | Computes the Non-negative Matrix Factorization using multiplicative update rules. |
| `ComputeReconstructionError(Matrix<>,Matrix<>,Matrix<>)` | Computes the Frobenius norm of the reconstruction error \|\|V - W * H\|\|. |
| `Decompose` | Performs the NMF decomposition. |
| `InitializeRandomMatrix(Int32,Int32,)` | Initializes a matrix with random positive values scaled to the data magnitude. |
| `Reconstruct` | Reconstructs the original matrix from the factorization. |
| `RunNmfTrial(Matrix<>,Int32,Int32,Int32,Int32,Double,)` | Runs a single trial of NMF with random initialization. |
| `Solve(Vector<>)` | Solves a linear system Ax = b using the NMF decomposition. |
| `SolveLinearSystem(Matrix<>,Vector<>)` | Solves a linear system using Gaussian elimination with partial pivoting. |

