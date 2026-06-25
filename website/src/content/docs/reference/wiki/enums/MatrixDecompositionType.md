---
title: "MatrixDecompositionType"
description: "Specifies different methods for breaking down (decomposing) matrices into simpler components."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies different methods for breaking down (decomposing) matrices into simpler components.

## For Beginners

Matrix decomposition is like breaking a complex puzzle into simpler pieces 
that are easier to work with. In mathematics, we often need to break down complex matrices 
(grids of numbers) into simpler components to solve problems more efficiently.

Think of it as:

- Breaking down a complex number like 15 into its factors 3 × 5
- Disassembling a complicated machine into its basic parts
- Converting a difficult problem into several easier ones

Matrix decompositions are important in AI and machine learning for:

- Solving systems of equations efficiently
- Reducing the dimensionality of data
- Finding patterns in data
- Making certain calculations faster and more stable
- Enabling specific types of analysis

Different decomposition methods have different strengths, weaknesses, and use cases.

## Fields

| Field | Summary |
|:-----|:--------|
| `Bidiagonal` | Transforms a matrix into a bidiagonal form (non-zero elements only on the main diagonal and either the diagonal above or below it). |
| `Cholesky` | A decomposition for symmetric, positive-definite matrices into a lower triangular matrix and its transpose. |
| `Cramer` | A method for solving systems of linear equations using determinants. |
| `Eigen` | Decomposes a matrix in terms of its eigenvalues and eigenvectors. |
| `GramSchmidt` | A method for converting a set of vectors into an orthogonal or orthonormal set. |
| `Hessenberg` | Transforms a matrix into Hessenberg form, which is nearly triangular. |
| `Ica` | Independent Component Analysis - separates mixed signals into statistically independent source components. |
| `Ldl` | Decomposes a symmetric matrix into the product L·D·Lᵀ, where L is lower triangular with 1s on the diagonal and D is diagonal. |
| `Lq` | A variant of QR decomposition that produces a lower triangular matrix L and an orthogonal matrix Q. |
| `Lu` | Decomposes a matrix into a product of lower and upper triangular matrices. |
| `Nmf` | Non-negative Matrix Factorization - decomposes a non-negative matrix into two non-negative matrices W and H. |
| `Normal` | Transforms a matrix into a form that simplifies certain calculations in least squares problems. |
| `Polar` | Decomposes a matrix into a product of a unitary matrix and a positive semi-definite Hermitian matrix. |
| `Qr` | Decomposes a matrix into an orthogonal matrix Q and an upper triangular matrix R. |
| `Schur` | Decomposes a matrix into a product involving an orthogonal matrix and a quasi-triangular matrix. |
| `Svd` | Singular Value Decomposition - factorizes a matrix into three components representing rotation, scaling, and another rotation. |
| `Takagi` | A decomposition for complex symmetric matrices. |
| `Tridiagonal` | Transforms a matrix into a tridiagonal form (non-zero elements only on the main diagonal and the diagonals above and below it). |
| `Udu` | Decomposes a symmetric matrix into the product U·D·Uᵀ, where U is upper triangular with 1s on the diagonal and D is diagonal. |

