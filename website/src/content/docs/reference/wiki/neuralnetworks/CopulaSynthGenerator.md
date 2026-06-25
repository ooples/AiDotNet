---
title: "CopulaSynthGenerator<T>"
description: "Copula-Based Synthesis generator that models marginal distributions independently and couples them with a Gaussian copula to capture inter-feature dependencies."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

Copula-Based Synthesis generator that models marginal distributions independently
and couples them with a Gaussian copula to capture inter-feature dependencies.

## For Beginners

This method works like building synthetic data from two ingredients:

Ingredient 1 — Marginal shapes:
Each feature's histogram is learned independently (e.g., "Age is bell-shaped around 40").

Ingredient 2 — Correlation structure:
How features move together (e.g., "Age and Income go up together").

To generate: sample correlated random values, then map each one to its feature's histogram.
The result preserves both individual distributions and pairwise correlations.

## How It Works

The generator operates in three phases:

1. Fit marginal distributions for each feature using empirical CDF / kernel density estimation
2. Transform data to uniform [0,1] via marginal CDFs, then to standard normal via inverse CDF
3. Fit a Gaussian copula (correlation matrix) on the normal-transformed data

To generate:

1. Sample from multivariate normal using the learned correlation matrix
2. Transform back to uniform via standard normal CDF
3. Transform to original scale via inverse marginal CDFs (quantile functions)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CopulaSynthGenerator(CopulaSynthOptions<>)` | Initializes a new instance of the `CopulaSynthGenerator` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CholeskyDecompose(Double[0:,0:],Int32)` | Performs Cholesky decomposition: A = L * L^T, returns L. |
| `ComputeCorrelation(Double[0:,0:],Int32,Int32)` | Computes the correlation matrix from normal-transformed data. |
| `EmpiricalCDF(Int32,Double)` | Forward empirical CDF for the probability-integral transform. |
| `Erf(Double)` | Error function approximation using Horner form. |
| `FitInternal(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` |  |
| `GenerateInternal(Int32,Vector<>,Vector<>)` |  |
| `InverseEmpiricalCDF(Int32,Double)` | Inverse empirical CDF (quantile function) on the Hazen grid that `Double)` uses: plotting position `(i + 0.5)/n` maps back to `sorted[i]`. |
| `InverseNormalCDF(Double)` | Inverse standard normal CDF using rational approximation (Beasley-Springer-Moro). |
| `LowerBound(Double[],Double)` | First index i with sorted[i] >= value (lower bound). |
| `NormalCDF(Double)` | Standard normal CDF using the error function approximation. |
| `UpperBound(Double[],Double)` | First index i with sorted[i] > value (upper bound). |

