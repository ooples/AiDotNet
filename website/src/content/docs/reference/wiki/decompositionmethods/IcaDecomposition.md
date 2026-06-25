---
title: "IcaDecomposition<T>"
description: "Implements Independent Component Analysis (ICA) for blind source separation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.MatrixDecomposition`

Implements Independent Component Analysis (ICA) for blind source separation.

## For Beginners

ICA is a technique used to separate mixed signals into their independent sources.
Imagine you're at a party with multiple people talking, and you have multiple microphones recording
the mixed conversations. ICA helps you separate out each individual voice.

## How It Works

The key idea is that ICA finds statistically independent components that were combined together.
This is different from other decomposition methods because it focuses on statistical independence
rather than uncorrelated patterns.

Common applications include:

- Audio source separation (cocktail party problem)
- Brain signal analysis (EEG, fMRI)
- Image separation and feature extraction
- Financial data analysis
- Telecommunications signal processing

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IcaDecomposition(Matrix<>,Nullable<Int32>,Int32,Double)` | Initializes a new instance of the ICA decomposition for the specified matrix. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IndependentComponents` | Gets the independent components (separated sources). |
| `Mean` | Gets the mean vector used for centering the data. |
| `MixingMatrix` | Gets the mixing matrix (inverse of unmixing matrix). |
| `UnmixingMatrix` | Gets the unmixing matrix (separation matrix). |
| `WhiteningMatrix` | Gets the whitening matrix used in the preprocessing step. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CenterData(Matrix<>,Vector<>)` | Centers the data by subtracting the mean from each column. |
| `ComputeColumnMean(Matrix<>)` | Computes the mean of each column in the matrix. |
| `ComputeFastIca(Matrix<>,Int32,Int32,Double)` | Computes Independent Component Analysis using the FastICA algorithm. |
| `ComputeMixingMatrix(Matrix<>,Matrix<>)` | Computes the mixing matrix from the unmixing matrix and whitening matrix. |
| `Decompose` | Performs the ICA decomposition. |
| `FastIcaAlgorithm(Matrix<>,Int32,Int32,Double)` | Implements the FastICA algorithm to find the unmixing matrix. |
| `GramSchmidtOrthogonalization(Matrix<>)` | Orthogonalizes matrix rows using the Gram-Schmidt process. |
| `Solve(Vector<>)` | Solves a linear system Ax = b using the ICA decomposition. |
| `Tanh()` | Computes the hyperbolic tangent (tanh) function using only INumericOperations. |
| `Transform(Matrix<>)` | Transforms new data using the learned ICA model. |
| `WhitenData(Matrix<>,Int32)` | Whitens the data using eigenvalue decomposition. |

