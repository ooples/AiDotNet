---
title: "FactorAnalysis<T>"
description: "Factor Analysis for dimensionality reduction with noise modeling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Factor Analysis for dimensionality reduction with noise modeling.

## For Beginners

Factor Analysis is like PCA but smarter about noise:

- PCA assumes all variance is signal
- Factor Analysis separates "common variance" (shared across features)

from "unique variance" (noise specific to each feature)

- Use when your features have different noise levels
- Popular in psychology, social sciences, and survey analysis

## How It Works

Factor Analysis assumes that the observed data X is generated from a set of
latent factors F plus feature-specific noise: X = F * W + noise.
Unlike PCA, Factor Analysis explicitly models unique variance (noise) for
each feature.

The model assumes:

- X = W * F + ε
- Where ε ~ N(0, Ψ) and Ψ is diagonal (unique variances)
- F ~ N(0, I) are the latent factors

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FactorAnalysis(Int32,Int32,Double,FactorRotation,Nullable<Int32>,Int32[])` | Creates a new instance of `FactorAnalysis`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Components` | Gets the factor loadings (each column is a factor). |
| `Mean` | Gets the mean of each feature. |
| `NComponents` | Gets the number of factors. |
| `NoiseVariance` | Gets the unique variance (noise) for each feature. |
| `Rotation` | Gets the rotation method. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits Factor Analysis using the EM algorithm. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported for Factor Analysis. |
| `TransformCore(Matrix<>)` | Transforms the data by computing factor scores. |

