---
title: "NormalDecomposition<T>"
description: "Implements the Normal Equation method for solving linear systems, especially useful for overdetermined systems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Implements the Normal Equation method for solving linear systems, especially useful for overdetermined systems.

## For Beginners

When you have more equations than unknowns (like having 10 data points but only 
wanting to find the slope and intercept of a line), this method helps find the "best fit" solution.
It works by converting your original problem into a simpler one that can be solved more easily.
Think of it like finding the average of several measurements - you're finding the solution that
minimizes the overall error.

## How It Works

The Normal Equation transforms a potentially non-square system Ax = b into a square system (A^T)Ax = (A^T)b,
which can then be solved using Cholesky decomposition for efficiency and stability.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NormalDecomposition(Matrix<>)` | Initializes a new instance of the NormalDecomposition class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `_aTA` | The product of A-transpose and A, forming a square, symmetric matrix. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Decompose` | Decomposition is performed in the constructor via Cholesky decomposition. |
| `Invert` | Calculates the inverse of the original matrix using the normal equation method. |
| `Solve(Vector<>)` | Solves the system Ax = b using the normal equation method. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_choleskyDecomposition` | The Cholesky decomposition of the A^T*A matrix, used to efficiently solve the system. |

