---
title: "ComplexMatrixDecomposition<T>"
description: "A wrapper class that adapts a real-valued matrix decomposition to work with complex numbers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

A wrapper class that adapts a real-valued matrix decomposition to work with complex numbers.

## For Beginners

This is a wrapper that lets you apply regular matrix decomposition methods
to complex numbers. Complex numbers have both real and imaginary parts (like 3 + 4i), but this
implementation currently works best when the imaginary parts are zero.

## How It Works

This class allows you to use existing matrix decomposition algorithms with complex numbers
by wrapping a real-valued decomposition. Note that this implementation only works with
matrices that have real values (the imaginary parts are all zero).

Real-world applications:

- Transitioning between real and complex number computations
- Testing complex-valued algorithms with real-valued data
- Quantum mechanics simulations with real-valued initial conditions

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ComplexMatrixDecomposition(IMatrixDecomposition<>)` | Creates a new complex matrix decomposition by wrapping a real-valued decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ConvertToComplexMatrix(Matrix<>,INumericOperations<>)` | Converts a real-valued matrix to a complex matrix with zero imaginary parts. |
| `Decompose` | Decomposition is handled by the wrapped base decomposition. |
| `Invert` | Calculates the inverse of the original matrix. |
| `Solve(Vector<Complex<>>)` | Solves a linear system of equations Ax = b, where A is the decomposed matrix. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_baseDecomposition` | The underlying real-valued matrix decomposition. |
| `_realOps` | Operations for the numeric type T (like addition, multiplication, etc.). |

