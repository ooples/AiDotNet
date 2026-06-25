---
title: "ProbabilisticPCA<T>"
description: "Probabilistic Principal Component Analysis (PPCA)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Probabilistic Principal Component Analysis (PPCA).

## For Beginners

PPCA extends standard PCA by:

- Providing a probabilistic model for the data
- Estimating noise variance σ² separately
- Enabling computation of data likelihoods
- Handling missing values naturally

Use cases:

- When you need uncertainty estimates
- Data with missing values
- Model comparison using likelihoods
- Bayesian machine learning pipelines

## How It Works

Probabilistic PCA is a probabilistic formulation of PCA that models data as being
generated from a lower-dimensional latent space with added Gaussian noise. This
provides a proper likelihood model, enables handling of missing data, and allows
for Bayesian extensions.

The model:
x = W * z + μ + ε
where z ~ N(0, I), ε ~ N(0, σ²I)

The algorithm:

1. Center the data
2. Use EM algorithm or closed-form solution to estimate W and σ²
3. Project data to latent space using posterior mean

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProbabilisticPCA(Int32,Int32,Double,Nullable<Int32>,Int32[])` | Creates a new instance of `ProbabilisticPCA`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LoadingMatrix` | Gets the loading matrix W. |
| `NComponents` | Gets the number of components (dimensions). |
| `NoiseVariance` | Gets the estimated noise variance. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits Probabilistic PCA using the EM algorithm. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reconstructs data from the latent space. |
| `Score(Matrix<>)` | Computes the log-likelihood of the data under the model. |
| `TransformCore(Matrix<>)` | Transforms data by projecting to the latent space. |

