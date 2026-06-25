---
title: "UduDecomposition<T>"
description: "Represents a UDU' decomposition of a matrix, which factorizes a symmetric matrix A into U*D*U', where U is an upper triangular matrix with ones on the diagonal, D is a diagonal matrix, and U' is the transpose of U."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Represents a UDU' decomposition of a matrix, which factorizes a symmetric matrix A into U*D*U',
where U is an upper triangular matrix with ones on the diagonal, D is a diagonal matrix,
and U' is the transpose of U.

## For Beginners

UDU' decomposition is a way to break down a complex matrix into simpler parts.
Think of it like factoring a number (e.g., 12 = 3 * 4). This decomposition is particularly
useful for solving systems of linear equations and for numerical stability in calculations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UduDecomposition(Matrix<>,UduAlgorithmType)` | Initializes a new instance of the UduDecomposition class and performs the decomposition. |

## Properties

| Property | Summary |
|:-----|:--------|
| `D` | Gets the diagonal matrix D represented as a vector of its diagonal elements. |
| `U` | Gets the upper triangular matrix U from the decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDecomposition(UduAlgorithmType)` | Computes the UDU' decomposition using the specified algorithm. |
| `Decompose` | Performs the UDU' decomposition. |
| `DecomposeCrout` | Performs the UDU' decomposition using the Crout algorithm. |
| `DecomposeDoolittle` | Performs the UDU' decomposition using the Doolittle algorithm. |
| `GetFactors` | Returns the U and D factors from the UDU' decomposition. |
| `Invert` | Calculates the inverse of the original matrix A using the UDU' decomposition. |
| `Solve(Vector<>)` | Solves the linear system Ax = b using the UDU' decomposition. |

