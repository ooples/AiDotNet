---
title: "SchurDecomposition<T>"
description: "Performs Schur decomposition on a matrix, factoring it into the product of a unitary matrix and an upper triangular matrix."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Performs Schur decomposition on a matrix, factoring it into the product of a unitary matrix and an upper triangular matrix.

## For Beginners

Schur decomposition breaks down a complex matrix into simpler parts that are easier to work with.
It's like factoring a number (e.g., 12 = 3 * 4), but for matrices. The decomposition produces two matrices:
a unitary matrix (which preserves lengths and angles) and an upper triangular matrix (which has zeros below the diagonal).
This makes many calculations much simpler.

## How It Works

Schur decomposition factors a matrix A into the product A = USU*, where U is a unitary matrix
and S is an upper triangular matrix. This decomposition is particularly useful for computing
eigenvalues and for analyzing the properties of linear transformations.

Real-world applications:

- Computing eigenvalues and eigenvectors efficiently
- Solving differential equations in engineering
- Control theory and system stability analysis

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SchurDecomposition(Matrix<>,SchurAlgorithmType)` | Initializes a new instance of the SchurDecomposition class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SchurMatrix` | Gets the upper triangular Schur matrix (S) from the decomposition. |
| `UnitaryMatrix` | Gets the unitary matrix (U) from the decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyFrancisDoubleShift(Matrix<>,Matrix<>,Int32)` | Applies a Francis double-shift QR step using Givens rotations. |
| `ComputeDecomposition(Matrix<>,SchurAlgorithmType)` | Computes the Schur decomposition using the specified algorithm. |
| `ComputeHouseholderReflection(,,)` | Computes a Householder reflection matrix based on the input vector components. |
| `ComputeSchurFrancis(Matrix<>)` | Computes the Schur decomposition using the Francis QR algorithm. |
| `ComputeSchurImplicit(Matrix<>)` | Computes the Schur decomposition using the implicit QR algorithm. |
| `ComputeSchurQR(Matrix<>)` | Computes the Schur decomposition using the QR algorithm. |
| `Decompose` | Performs the Schur decomposition. |
| `Invert` | Computes the inverse of the original matrix using the Schur decomposition. |
| `Solve(Vector<>)` | Solves a system of linear equations Ax = b using the Schur decomposition. |

