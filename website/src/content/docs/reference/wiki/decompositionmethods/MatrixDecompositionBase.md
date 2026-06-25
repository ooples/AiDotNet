---
title: "MatrixDecompositionBase<T>"
description: "Base class for matrix decomposition algorithms that break down matrices into simpler components."
section: "API Reference"
---

`Base Classes` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Base class for matrix decomposition algorithms that break down matrices into simpler components.

## For Beginners

Matrix decomposition is like factoring a number, but for matrices.
Just as we can break down 12 into 3 * 4, we can break down complex matrices into simpler
matrices that are easier to work with. Different decomposition methods have different strengths
and are used for different purposes in machine learning and numerical computing.

## How It Works

This base class provides common functionality that all matrix decompositions need, such as:

- Storing the original matrix
- Providing numeric operations
- Implementing matrix inversion
- Solving linear systems

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MatrixDecompositionBase(Matrix<>)` | Initializes a new instance of the matrix decomposition class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `A` | The original matrix that was decomposed. |
| `Engine` | Gets the global execution engine for vector operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApproximatelyEqual(,,)` | Checks if two values are approximately equal within a tolerance. |
| `Decompose` | Performs the actual decomposition of the matrix into its components. |
| `FrobeniusNorm(Matrix<>)` | Computes the Frobenius norm of a matrix (square root of sum of squared elements). |
| `Invert` | Calculates the inverse of the original matrix A using the decomposition. |
| `Solve(Vector<>)` | Solves a linear system of equations Ax = b, where A is the decomposed matrix. |
| `ValidateMatrix(Matrix<>,Boolean,Boolean,Boolean)` | Validates that the matrix meets the requirements for this decomposition method. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides mathematical operations for the numeric type T. |

