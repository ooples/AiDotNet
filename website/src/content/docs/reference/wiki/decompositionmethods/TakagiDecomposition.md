---
title: "TakagiDecomposition<T>"
description: "Implements the Takagi factorization for complex symmetric matrices."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Implements the Takagi factorization for complex symmetric matrices.

## For Beginners

The Takagi decomposition is a special matrix factorization that works on symmetric matrices.
It breaks down a matrix into simpler components that make calculations easier. Think of it like
factoring a number into its prime components, but for matrices.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TakagiDecomposition(Matrix<>,TakagiAlgorithmType)` | Initializes a new instance of the TakagiDecomposition class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SigmaMatrix` | Gets the diagonal matrix containing the singular values. |
| `UnitaryMatrix` | Gets the unitary matrix used in the decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateMagnitude(Complex<>)` | Calculates the magnitude (absolute value) of a complex number. |
| `ComputeDecomposition(Matrix<>,TakagiAlgorithmType)` | Computes the Takagi decomposition using the specified algorithm. |
| `ComputeTakagiDefault(Matrix<>)` | Computes the Takagi decomposition using a default approach based on eigendecomposition. |
| `ComputeTakagiEigenDecomposition(Matrix<>)` | Computes the Takagi decomposition using eigendecomposition. |
| `ComputeTakagiJacobi(Matrix<>)` | Computes the Takagi decomposition using the Jacobi algorithm. |
| `ComputeTakagiLanczosIteration(Matrix<>)` | Computes the Takagi decomposition using the Lanczos iteration method. |
| `ComputeTakagiPowerIteration(Matrix<>)` | Computes the Takagi decomposition using the power iteration method. |
| `ComputeTakagiQR(Matrix<>)` | Computes the Takagi decomposition using the QR algorithm. |
| `Decompose` | Performs the Takagi decomposition. |
| `GetMachineEpsForType` | Type-appropriate machine epsilon. |
| `Invert` | Inverts the matrix using the Takagi decomposition. |
| `QRDecomposition(Matrix<>)` | Performs QR decomposition on a matrix. |
| `Solve(Vector<>)` | Solves a linear system of equations Ax = b, where A is the matrix represented by this decomposition. |
| `SumAbsUpperOffDiagonal(Matrix<>)` | Sum of `\|a[i,j]\|` over the strict upper triangle (i < j). |
| `SumSquaredEntries(Matrix<>)` | Sum of squared entries — used for the Frobenius-norm-based convergence threshold in the cyclic-Jacobi sweep loop. |
| `SumSquaredOffDiagonal(Matrix<>)` | Sum of squared off-diagonal entries (i ≠ j). |

