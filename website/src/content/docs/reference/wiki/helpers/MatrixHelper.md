---
title: "MatrixHelper<T>"
description: "Provides helper methods for matrix operations used in AI and machine learning algorithms."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides helper methods for matrix operations used in AI and machine learning algorithms.

## For Beginners

A matrix is a rectangular array of numbers arranged in rows and columns.
Matrices are fundamental in machine learning for representing data, transformations, and 
mathematical operations.

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGivensRotation(Matrix<>,,,Int32,Int32,Int32,Int32)` | Applies a Givens rotation to specific rows of a matrix. |
| `ApplyHouseholderTransformation(Matrix<>,Vector<>,Int32)` | Applies a Householder transformation to a matrix. |
| `BandDiagonalMultiply(Int32,Int32,Matrix<>,Vector<>,Vector<>)` | Multiplies a band diagonal matrix by a vector. |
| `CalculateDeterminantRecursive(Matrix<>)` | Calculates the determinant of a matrix using a recursive algorithm. |
| `CalculateHatMatrix(Matrix<>)` | Calculates the Hat Matrix (also known as the projection matrix) used in regression analysis. |
| `ComputeGivensRotation(,)` | Computes the cosine and sine components of a Givens rotation. |
| `CreateHouseholderVector(Vector<>)` | Creates a Householder vector from a given vector. |
| `CreateSubMatrix(Matrix<>,Int32,Int32)` | Creates a submatrix by excluding a specified row and column from the original matrix. |
| `ExtractDiagonal(Matrix<>)` | Extracts the diagonal elements of a matrix into a vector. |
| `Hypotenuse(,)` | Calculates the hypotenuse of a right triangle given the lengths of the other two sides. |
| `Hypotenuse([])` | Calculates the Euclidean norm (magnitude) of a vector of values. |
| `InvertUsingDecomposition(IMatrixDecomposition<>)` | Inverts a matrix using a provided matrix decomposition. |
| `IsInvertible(Matrix<>)` | Determines if a matrix is invertible (non-singular). |
| `IsUpperHessenberg(Matrix<>,)` | Determines if a matrix is in upper Hessenberg form within a specified tolerance. |
| `OrthogonalizeColumns(Matrix<>)` | Orthogonalizes the columns of a matrix using the Gram-Schmidt process. |
| `OuterProduct(Vector<>,Vector<>)` | Computes the outer product of two vectors. |
| `PowerIteration(Matrix<>,Int32,)` | Implements the power iteration algorithm to find the dominant eigenvalue and eigenvector of a matrix. |
| `ReduceToHessenbergFormat(Matrix<>)` | Reduces a matrix to Hessenberg form, which is useful for eigenvalue calculations. |
| `SpectralNorm(Matrix<>)` | Calculates the spectral norm of a matrix, which is the largest singular value. |
| `TridiagonalSolve(Vector<>,Vector<>,Vector<>,Vector<>,Vector<>)` | Solves a tridiagonal system of linear equations. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Provides operations for the numeric type T. |

