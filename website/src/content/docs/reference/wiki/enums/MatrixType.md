---
title: "MatrixType"
description: "Defines the different types of matrices that can be used in mathematical operations."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the different types of matrices that can be used in mathematical operations.

## For Beginners

A matrix is a rectangular array of numbers arranged in rows and columns.
Different types of matrices have special properties that make them useful for specific
calculations or applications. This enum lists the various matrix types supported by the library.

## Fields

| Field | Summary |
|:-----|:--------|
| `Adjacency` | A matrix that represents connections in a graph or network. |
| `Band` | A matrix with non-zero elements only on a band centered on the main diagonal. |
| `Block` | A matrix divided into submatrices (blocks) that are treated as single elements. |
| `Cauchy` | A matrix where each element is 1 divided by the sum of two values from separate arrays. |
| `Circulant` | A special Toeplitz matrix where each row is a cyclic shift of the row above it. |
| `Companion` | A special matrix used in polynomial calculations and control theory. |
| `Dense` | A matrix where most elements are non-zero. |
| `Diagonal` | A matrix where all non-diagonal elements are zero. |
| `DoublyStochastic` | A matrix where all elements are non-negative and both rows and columns sum to 1. |
| `Hankel` | A matrix where each anti-diagonal (running from bottom-left to top-right) has constant values. |
| `Hermitian` | A complex square matrix that equals its own conjugate transpose. |
| `Hilbert` | A special matrix where each element is 1 divided by the sum of its row and column indices. |
| `Idempotent` | A matrix that, when multiplied by itself, gives the same matrix. |
| `Identity` | A diagonal matrix where all diagonal elements are 1. |
| `Incidence` | A matrix that shows relationships between two types of objects in a graph. |
| `Involutory` | A matrix that, when multiplied by itself, gives the identity matrix. |
| `Laplacian` | A matrix that represents a graph's connectivity and is used in spectral graph theory. |
| `LowerBidiagonal` | A matrix with non-zero elements only on the main diagonal and the diagonal below it. |
| `LowerTriangular` | A matrix where all elements above the main diagonal are zero. |
| `NonSingular` | A square matrix that has an inverse. |
| `Orthogonal` | A real square matrix whose transpose equals its inverse. |
| `OrthogonalProjection` | A square matrix that represents a projection onto a subspace. |
| `Partitioned` | A matrix that has been divided into sections for specific mathematical operations. |
| `Permutation` | A matrix that has exactly one 1 in each row and each column, with all other elements being 0. |
| `PositiveDefinite` | A symmetric matrix where all eigenvalues are positive. |
| `PositiveSemiDefinite` | A symmetric matrix where all eigenvalues are non-negative (zero or positive). |
| `Rectangular` | A matrix with a different number of rows and columns. |
| `Scalar` | A diagonal matrix where all diagonal elements are the same value. |
| `Singular` | A square matrix that doesn't have an inverse. |
| `SkewHermitian` | A complex square matrix whose conjugate transpose equals its negative. |
| `SkewSymmetric` | A square matrix whose transpose equals its negative. |
| `Sparse` | A matrix where most elements are zero. |
| `Square` | A matrix with the same number of rows and columns. |
| `Stochastic` | A matrix where all elements are non-negative and each row sums to 1. |
| `Symmetric` | A square matrix that is equal to its transpose (mirror image across the diagonal). |
| `Toeplitz` | A matrix where each descending diagonal from left to right has constant values. |
| `Tridiagonal` | A matrix with non-zero elements only on the main diagonal and the diagonals immediately above and below it. |
| `Unitary` | A complex square matrix whose conjugate transpose equals its inverse. |
| `Unknown` | Matrix type has not been determined or specified. |
| `UpperBidiagonal` | A matrix with non-zero elements only on the main diagonal and the diagonal above it. |
| `UpperTriangular` | A matrix where all elements below the main diagonal are zero. |
| `Vandermonde` | A matrix where each row consists of consecutive powers of a value. |
| `Zero` | A matrix where all elements are zero. |

