---
title: "LuDecomposition<T>"
description: "Implements LU decomposition for matrices, which factorizes a matrix into a product of lower and upper triangular matrices."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Implements LU decomposition for matrices, which factorizes a matrix into a product of lower and upper triangular matrices.

## For Beginners

LU decomposition breaks a matrix into two simpler triangular matrices:
L (lower triangular, with values only on and below the diagonal) and U (upper triangular, with
values only on and above the diagonal). Think of it like factoring a number into simpler parts,
but for matrices. This makes solving equations much faster.

## How It Works

LU decomposition factors a matrix A into the product of a lower triangular matrix L and an upper
triangular matrix U, often with row permutations represented by a permutation matrix P (P*A = L*U).
This decomposition is fundamental in numerical linear algebra and is used for solving linear systems,
computing determinants, and matrix inversion.

Real-world applications:

- Solving systems of linear equations efficiently
- Computing matrix determinants
- Finding matrix inverses in numerical algorithms

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LuDecomposition(Matrix<>,LuAlgorithmType)` | Initializes a new instance of the LuDecomposition class and performs the decomposition. |

## Properties

| Property | Summary |
|:-----|:--------|
| `L` | Gets the lower triangular matrix from the decomposition. |
| `P` | Gets the permutation vector that tracks row exchanges during pivoting. |
| `U` | Gets the upper triangular matrix from the decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BackSubstitution(Matrix<>,Vector<>)` | Performs back substitution to solve the system Ux = y where U is upper triangular. |
| `ComputeCholesky(Matrix<>)` | Computes the Cholesky decomposition of a symmetric positive-definite matrix. |
| `ComputeDecomposition(Matrix<>,LuAlgorithmType)` | Performs the matrix decomposition using the specified algorithm. |
| `ComputeLuCrout(Matrix<>)` | Computes the LU decomposition using Crout's method. |
| `ComputeLuDoolittle(Matrix<>)` | Computes the LU decomposition using Doolittle's method. |
| `ComputeLuPartialPivoting(Matrix<>)` | Computes LU decomposition with partial pivoting (Gaussian elimination with row pivoting). |
| `Decompose` | Performs the LU decomposition. |
| `ForwardSubstitution(Matrix<>,Vector<>)` | Performs forward substitution to solve the system Ly = b where L is lower triangular. |
| `PermutateVector(Vector<>,Vector<Int32>)` | Rearranges a vector according to the permutation vector P. |
| `Solve(Vector<>)` | Solves the linear system Ax = b using the LU decomposition. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_algorithm` | The algorithm to use for LU decomposition. |

