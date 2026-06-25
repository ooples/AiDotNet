---
title: "InverseType"
description: "Specifies different algorithms for calculating matrix inverses in mathematical operations."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies different algorithms for calculating matrix inverses in mathematical operations.

## For Beginners

A matrix inverse is like finding the opposite of a number. Just as 1/5 is the 
inverse of 5 (because 5 × 1/5 = 1), a matrix inverse is a special matrix that, when multiplied 
with the original matrix, gives the identity matrix (the matrix equivalent of the number 1).

Matrix inverses are important in AI and machine learning for:

- Solving systems of equations
- Finding optimal parameters in linear regression
- Transforming data
- Many other mathematical operations

Different algorithms for finding inverses have different trade-offs in terms of speed, 
accuracy, and memory usage. This enum lets you choose which algorithm to use.

## Fields

| Field | Summary |
|:-----|:--------|
| `GaussianJordan` | A direct method for finding matrix inverses using elementary row operations. |
| `Newton` | An iterative algorithm that approximates the inverse through successive refinements. |
| `Strassen` | A divide-and-conquer algorithm for matrix inversion that's efficient for large matrices. |

