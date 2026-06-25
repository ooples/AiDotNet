---
title: "VectorExtensions"
description: "Provides extension methods for vector operations commonly used in AI and machine learning applications."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Extensions`

Provides extension methods for vector operations commonly used in AI and machine learning applications.

## For Beginners

A vector in AI is simply a list of numbers that can represent data points, 
features, or weights in a model. These extension methods provide ways to manipulate and 
perform calculations on these lists of numbers.

## Methods

| Method | Summary |
|:-----|:--------|
| `AbsoluteMaximum(Vector<>)` | Finds the element with the largest absolute value in the vector. |
| `Add(Vector<>,)` | Adds a scalar value to each element of the vector. |
| `Add(Vector<>,Vector<>)` | Adds two vectors element by element. |
| `Argsort(Vector<>)` | Returns the indices that would sort the vector in ascending order. |
| `Average(Vector<>)` | Calculates the average (mean) of all elements in the vector. |
| `CreateDiagonal(Vector<>)` | Creates a diagonal matrix from a vector. |
| `Divide(Vector<>,)` | Divides each element of the vector by a scalar value. |
| `DotProduct(Vector<>,Vector<>)` | Calculates the dot product (scalar product) of two vectors. |
| `EuclideanDistance(Vector<>,Vector<>)` | Calculates the Euclidean distance between two vectors. |
| `Extract(Vector<>,Int32)` | Creates a new vector containing the first specified number of elements from the original vector. |
| `Magnitude(Vector<>)` | Calculates the magnitude (length) of a vector. |
| `Max(Vector<>)` | Finds the maximum value in the vector. |
| `MaxIndex(Vector<>)` | Finds the index of the maximum value in the vector. |
| `Maximum(Vector<>,)` | Creates a new vector where each element is the maximum of the corresponding element in the input vector and a scalar value. |
| `Median(Vector<>)` | Calculates the median value of the elements in the vector. |
| `Min(Vector<>)` | Finds the minimum value in the vector. |
| `MinIndex(Vector<>)` | Finds the index of the minimum value in the vector. |
| `Minimum(Vector<>)` | Finds the minimum element value in the vector. |
| `Multiply(Vector<>,)` | Multiplies each element of the vector by a scalar value. |
| `Multiply(Vector<>,Matrix<>)` | Multiplies a vector by a matrix, performing a vector-matrix multiplication. |
| `Norm(Vector<>)` | Calculates the Euclidean norm (magnitude or length) of the vector. |
| `OuterProduct(Vector<>,Vector<>)` | Computes the outer product of two vectors, resulting in a matrix. |
| `PointwiseAbs(Vector<>)` | Creates a new vector where each element is the absolute value of the corresponding element in the input vector. |
| `PointwiseDivide(Vector<>,Vector<>)` | Divides each element of the left vector by the corresponding element of the right vector. |
| `PointwiseExp(Vector<>)` | Applies the exponential function to each element of the vector. |
| `PointwiseLog(Vector<>)` | Applies the natural logarithm function to each element of the vector. |
| `PointwiseMultiply(Vector<>,Vector<>)` | Multiplies corresponding elements of two vectors together (Hadamard product). |
| `PointwiseMultiplyInPlace(Vector<>,Vector<>)` | Multiplies each element of the left vector by the corresponding element of the right vector and stores the result in the left vector. |
| `PointwiseSign(Vector<>)` | Returns a new vector where each element is the sign of the corresponding element in the input vector. |
| `PointwiseSqrt(Vector<>)` | Creates a new vector where each element is the square root of the corresponding element in the input vector. |
| `Repeat(Vector<>,Int32)` | Creates a new vector by repeating the original vector a specified number of times. |
| `Reshape(Vector<>,Int32,Int32)` | Reshapes a vector into a two-dimensional matrix with the specified number of rows and columns. |
| `Slice(Vector<>,Int32,Int32)` | Creates a new vector containing a subset of elements from the original vector. |
| `StandardDeviation(Vector<>)` | Calculates the standard deviation of the elements in the vector. |
| `SubVector(Vector<>,Int32,Int32)` | Creates a new vector containing a subset of elements from the original vector. |
| `SubVector(Vector<>,Int32[])` | Creates a new vector by selecting elements from the original vector at specified indices. |
| `Subtract(Vector<>,)` | Subtracts a scalar value from each element of the vector. |
| `Subtract(Vector<>,Vector<>)` | Subtracts the elements of the right vector from the corresponding elements of the left vector. |
| `Subvector(Vector<>,Int32[])` | Creates a subvector from the given vector using the specified indices. |
| `Sum(Vector<>)` | Calculates the sum of all elements in the vector. |
| `ToColumnMatrix(Vector<>)` | Converts the vector to a column matrix (n x 1). |
| `ToDiagonalMatrix(Vector<>)` | Converts the vector to a diagonal matrix. |
| `ToIntList(IEnumerable<Vector<>>)` | Converts a collection of vectors to a list of integers. |
| `ToRealVector(Vector<Complex<>>)` | Extracts the real parts from a vector of complex numbers. |
| `ToRowMatrix(Vector<>)` | Converts the vector to a row matrix (1 x n). |
| `ToVectorList(IEnumerable<Int32>)` | Converts a collection of integer indices to a list of single-element vectors. |
| `Transform(Vector<>,Func<,>)` | Applies a function to each element of the vector and returns a new vector with the results. |
| `Transform(Vector<>,Func<,>)` | Applies a function to each element of the vector and returns a new vector with the results, allowing for a change in the element type. |

