---
title: "BidiagonalAlgorithmType"
description: "Represents different algorithm types for bidiagonal matrix decomposition."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums.AlgorithmTypes`

Represents different algorithm types for bidiagonal matrix decomposition.

## For Beginners

Bidiagonal decomposition is a technique used in linear algebra and machine learning 
to simplify complex matrices (tables of numbers) into a special form that makes further calculations easier.

A bidiagonal matrix is a special type of matrix where non-zero values appear only on the main diagonal 
(from top-left to bottom-right) and either just above or just below this diagonal. All other values are zero.

This decomposition is often used as a step in solving systems of equations, finding eigenvalues, 
or performing Singular Value Decomposition (SVD), which is crucial for many machine learning algorithms 
like Principal Component Analysis (PCA), recommendation systems, and image compression.

Think of it like simplifying a complex recipe into basic steps that are easier to follow - 
the bidiagonal form makes complex matrix operations more manageable.

This enum lists different mathematical approaches to perform this decomposition, each with its own 
advantages in terms of accuracy, speed, or memory usage.

## Fields

| Field | Summary |
|:-----|:--------|
| `Givens` | Uses Givens rotations to transform a matrix into bidiagonal form. |
| `Householder` | Uses Householder reflections to transform a matrix into bidiagonal form. |
| `Lanczos` | Uses the Lanczos algorithm to transform a matrix into bidiagonal form. |

