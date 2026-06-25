---
title: "PolarDecomposition<T>"
description: "Implements the Polar Decomposition of a matrix, which factors a matrix A into the product of an orthogonal matrix U and a positive semi-definite matrix P."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Implements the Polar Decomposition of a matrix, which factors a matrix A into the product of
an orthogonal matrix U and a positive semi-definite matrix P.

## For Beginners

Think of Polar Decomposition as breaking down a transformation into two simpler steps:
first a rotation/reflection (U), and then a stretching/scaling (P). This is similar to how polar
coordinates break down a point into an angle and a distance.

## How It Works

The Polar Decomposition expresses a matrix A as A = UP, where U is orthogonal (U^T * U = I)
and P is positive semi-definite.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PolarDecomposition(Matrix<>,PolarAlgorithmType)` | Initializes a new instance of the PolarDecomposition class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `P` | Gets the positive semi-definite factor of the decomposition. |
| `U` | Gets the orthogonal factor of the decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDecomposition(PolarAlgorithmType)` | Computes the polar decomposition using the specified algorithm. |
| `Decompose` | Performs the polar decomposition. |
| `DecomposeHalleyIteration` | Performs polar decomposition using Halley's iterative method. |
| `DecomposeNewtonSchulz` | Performs polar decomposition using the Newton-Schulz iterative algorithm. |
| `DecomposeQRIteration` | Performs polar decomposition using QR iteration. |
| `DecomposeSVD` | Performs polar decomposition using Singular Value Decomposition (SVD). |
| `DecomposeScalingAndSquaring` | Performs polar decomposition using the scaling and squaring method. |
| `GetFactors` | Returns the factors U and P of the polar decomposition. |
| `Invert` | Computes the inverse of the original matrix A using its polar decomposition. |
| `Solve(Vector<>)` | Solves the linear system Ax = b using the polar decomposition. |

