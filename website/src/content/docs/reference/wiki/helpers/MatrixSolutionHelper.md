---
title: "MatrixSolutionHelper"
description: "Provides methods for solving linear systems of equations using various matrix decomposition techniques."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides methods for solving linear systems of equations using various matrix decomposition techniques.

## For Beginners

A linear system is a collection of linear equations with the same variables.
For example: 2x + 3y = 5 and 4x - y = 1 form a linear system. In matrix form, this is written as Ax = b,
where A is the coefficient matrix, x is the vector of variables we're solving for, and b is the vector of constants.
This helper class provides different ways to solve for x given A and b.

## Methods

| Method | Summary |
|:-----|:--------|
| `SolveCramer(Matrix<>,Vector<>)` | Solves a linear system using Cramer's rule. |
| `SolveEigen(Matrix<>,Vector<>)` | Solves a linear system using Eigenvalue decomposition. |
| `SolveGramSchmidt(Matrix<>,Vector<>)` | Solves a linear system using the Gram-Schmidt decomposition. |
| `SolveHessenberg(Matrix<>,Vector<>)` | Solves a linear system using Hessenberg decomposition. |
| `SolveLinearSystem(Matrix<>,Vector<>,MatrixDecompositionType)` | Solves a linear system of equations Ax = b using the specified decomposition method. |
| `SolveLinearSystem(Vector<>,IMatrixDecomposition<>)` | Solves a linear system using a pre-computed matrix decomposition. |
| `SolveNormal(Matrix<>,Vector<>)` | Solves a linear system using the normal equations approach. |
| `SolveSchur(Matrix<>,Vector<>)` | Solves a linear system using Schur decomposition. |

