---
title: "TridiagonalAlgorithmType"
description: "Represents different algorithm types for converting a matrix to tridiagonal form."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums.AlgorithmTypes`

Represents different algorithm types for converting a matrix to tridiagonal form.

## For Beginners

A tridiagonal matrix is a special type of square matrix where non-zero values appear only 
on the main diagonal and the diagonals directly above and below it. All other elements are zero.

For example, a 5×5 tridiagonal matrix looks like this (where * represents non-zero values):

* * 0 0 0
* * * 0 0

0 * * * 0
0 0 * * *
0 0 0 * *

Converting a matrix to tridiagonal form is an important step in many numerical algorithms, especially 
when finding eigenvalues and eigenvectors. It simplifies the original problem by transforming a dense 
matrix (with many non-zero elements) into a simpler form that's easier to work with.

This process is like simplifying a complex equation before solving it - the answer remains the same, 
but the work becomes much easier.

## Fields

| Field | Summary |
|:-----|:--------|
| `Givens` | Uses Givens rotations to convert a matrix to tridiagonal form. |
| `Householder` | Uses Householder reflections to convert a matrix to tridiagonal form. |
| `Lanczos` | Uses the Lanczos algorithm to convert a matrix to tridiagonal form, particularly efficient for large, sparse matrices. |

