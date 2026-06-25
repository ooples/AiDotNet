---
title: "CramerDecomposition<T>"
description: "Implements Cramer's rule for solving systems of linear equations and matrix inversion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Implements Cramer's rule for solving systems of linear equations and matrix inversion.

## For Beginners

Cramer's Rule is a formula-based method for solving systems of equations.
It uses determinants (a special number calculated from a matrix) to find the solution directly.
Think of it like using a formula to solve a math problem rather than using step-by-step algebra.
While elegant, it becomes slow for large matrices.

## How It Works

Cramer's rule is a method for solving systems of linear equations using determinants.
It works by replacing columns in the coefficient matrix with the solution vector
and calculating ratios of determinants. This method is primarily educational and not
recommended for large matrices due to its computational complexity.

Real-world applications:

- Solving small systems of equations (2x2 or 3x3) in engineering
- Teaching linear algebra concepts
- Theoretical analysis and mathematical proofs

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CramerDecomposition(Matrix<>)` | Creates a new Cramer's rule decomposition for the specified matrix. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Cofactor(Matrix<>,Int32,Int32)` | Calculates the cofactor of a matrix element at the specified position. |
| `Decompose` | Decompose is not applicable for Cramer's rule as it directly solves without factorization. |
| `Determinant(Matrix<>)` | Calculates the determinant of a matrix recursively using cofactor expansion. |
| `Invert` | Calculates the inverse of the original matrix using the adjugate method. |
| `ReplaceColumn(Matrix<>,Vector<>,Int32)` | Creates a copy of a matrix with one column replaced by a vector. |
| `Solve(Vector<>)` | Solves a system of linear equations Ax = b using Cramer's rule. |

