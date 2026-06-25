---
title: "GramSchmidtDecomposition<T>"
description: "Implements the Gram-Schmidt orthogonalization process to decompose a matrix into an orthogonal matrix Q and an upper triangular matrix R."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Implements the Gram-Schmidt orthogonalization process to decompose a matrix into an orthogonal matrix Q and an upper triangular matrix R.

## For Beginners

This class takes a matrix and breaks it down into two special matrices:
Q (a matrix with perpendicular columns) and R (an upper triangular matrix with values only on and above the diagonal).
Think of it like organizing a messy set of vectors into a neat, perpendicular coordinate system.
Together, these matrices can be multiplied to get back the original matrix: A = Q * R.

## How It Works

The Gram-Schmidt process transforms a set of vectors into a set of orthogonal vectors (vectors that are perpendicular to each other).
This decomposition is useful for solving linear systems, computing least squares solutions, and other numerical applications.
The result is a QR factorization where A = Q * R.

Real-world applications:

- Solving systems of linear equations
- Computing least squares solutions in regression analysis
- Numerical stability improvements in various algorithms

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GramSchmidtDecomposition(Matrix<>,GramSchmidtAlgorithmType)` | Creates a new Gram-Schmidt decomposition for the specified matrix. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Q` | Gets the orthogonal matrix Q from the decomposition. |
| `R` | Gets the upper triangular matrix R from the decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BackSubstitution(Matrix<>,Vector<>)` | Performs back-substitution to solve an upper triangular system Rx = y. |
| `ComputeClassicalGramSchmidt(Matrix<>)` | Computes the QR decomposition using the Classical Gram-Schmidt algorithm. |
| `ComputeDecomposition(Matrix<>,GramSchmidtAlgorithmType)` | Selects and applies the appropriate Gram-Schmidt algorithm based on the specified type. |
| `ComputeModifiedGramSchmidt(Matrix<>)` | Computes the QR decomposition using the Modified Gram-Schmidt algorithm. |
| `Decompose` | Performs the Gram-Schmidt decomposition. |
| `Invert` | Calculates the inverse of the original matrix A using the QR decomposition. |
| `Solve(Vector<>)` | Solves a system of linear equations Ax = b using the QR decomposition. |

