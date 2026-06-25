---
title: "EigenAlgorithmType"
description: "Represents different algorithm types for computing eigenvalues and eigenvectors of matrices."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums.AlgorithmTypes`

Represents different algorithm types for computing eigenvalues and eigenvectors of matrices.

## For Beginners

Eigenvalues and eigenvectors are special numbers and vectors associated with a matrix 
that help us understand the matrix's fundamental properties and behavior.

Think of a matrix as a transformation that changes the position, scale, or rotation of points in space. 
Most vectors will change both their direction and length when this transformation is applied. However, 
eigenvectors are special vectors that only change in length (but keep their direction) when the 
transformation is applied. The eigenvalue tells us how much the eigenvector is stretched or compressed.

Why are these important in AI and machine learning?

1. Principal Component Analysis (PCA): A popular technique for dimensionality reduction that uses 

eigenvectors to find the most important features in your data.

2. Recommendation Systems: Eigenvalue methods help identify patterns in user preferences.

3. Image Processing: Facial recognition and image compression often use eigenvalue techniques.

4. Natural Language Processing: Some algorithms use eigenvalues to analyze relationships between words.

This enum lists different mathematical approaches to find these eigenvalues and eigenvectors, each with 
its own advantages depending on the specific problem you're solving.

## Fields

| Field | Summary |
|:-----|:--------|
| `Jacobi` | Uses the Jacobi eigenvalue algorithm to find all eigenvalues and eigenvectors. |
| `PowerIteration` | Uses the power iteration method to find the dominant eigenvalue and eigenvector. |
| `QR` | Uses QR decomposition to find eigenvalues and eigenvectors. |

