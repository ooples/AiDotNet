---
title: "HessenbergDecomposition<T>"
description: "Implements Hessenberg decomposition, which transforms a matrix into a form that is almost triangular."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Implements Hessenberg decomposition, which transforms a matrix into a form that is almost triangular.

## For Beginners

Think of Hessenberg decomposition as a way to simplify a matrix by making
most elements below the diagonal equal to zero, which makes further calculations much faster.
It's like organizing a messy room by putting everything in its place - the matrix becomes
easier to work with even though it's not completely triangular.

## How It Works

A Hessenberg matrix has zeros below the first subdiagonal, making it easier to work with for
many numerical algorithms. This decomposition is often used as a preprocessing step for
eigenvalue calculations and solving linear systems.

Real-world applications:

- Preprocessing for eigenvalue computation
- Accelerating iterative methods for solving linear systems
- Control theory and system analysis

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HessenbergDecomposition(Matrix<>,HessenbergAlgorithmType)` | Initializes a new instance of the Hessenberg decomposition for the specified matrix. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HessenbergMatrix` | Gets the resulting Hessenberg matrix after decomposition. |
| `OrthogonalMatrix` | Gets the orthogonal transformation matrix Q from the decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDecomposition(Matrix<>,HessenbergAlgorithmType)` | Computes the Hessenberg decomposition using the specified algorithm. |
| `ComputeHessenbergElementaryTransformations(Matrix<>)` | Computes the Hessenberg form using elementary row transformations. |
| `ComputeHessenbergGivens(Matrix<>)` | Computes the Hessenberg form using Givens rotations. |
| `ComputeHessenbergHouseholder(Matrix<>)` | Computes the Hessenberg form using Householder reflections. |
| `ComputeHessenbergImplicitQR(Matrix<>)` | Computes the Hessenberg form using the implicit QR algorithm. |
| `ComputeHessenbergLanczos(Matrix<>)` | Computes the Hessenberg form using the Lanczos algorithm. |
| `Decompose` | Performs the Hessenberg decomposition. |
| `Invert` | Calculates the inverse of the original matrix using the Hessenberg decomposition. |
| `Solve(Vector<>)` | Solves a linear system Ax = b using the Hessenberg decomposition. |
| `SolveHessenbergSystem(Matrix<>,Vector<>)` | Solves a Hessenberg system H*x = b using Gaussian elimination. |

