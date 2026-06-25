---
title: "CausalDiscoveryBase<T>"
description: "Abstract base class for causal discovery algorithms with shared statistical utilities."
section: "API Reference"
---

`Base Classes` · `AiDotNet.CausalDiscovery`

Abstract base class for causal discovery algorithms with shared statistical utilities.

## For Beginners

This base class contains the "toolbox" of statistical tests
and helper methods that all causal discovery algorithms share. Each specific algorithm
(like NOTEARS or PC) extends this class and adds its own unique logic.

## How It Works

Provides common functionality needed by most causal discovery algorithms:
conditional independence testing, BIC scoring, partial correlation computation,
covariance estimation, and OLS regression.

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `Engine` | Gets the global execution engine for accelerated vector/matrix operations. |
| `Name` |  |
| `SupportsLatentConfounders` |  |
| `SupportsMixedData` |  |
| `SupportsNonlinear` |  |
| `SupportsTimeSeries` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeBICScore(Matrix<>,Int32,Int32[])` | Computes the BIC (Bayesian Information Criterion) score for a variable given its parents. |
| `ComputeCovarianceMatrix(Matrix<>)` | Computes the sample covariance matrix from data. |
| `ComputePartialCorrelation(Matrix<>,Int32,Int32,Int32[])` | Computes the partial correlation between variables i and j given a conditioning set. |
| `ComputeResidualVariance(Matrix<>,Int32,Int32[])` | Fits OLS regression and returns the residual variance. |
| `ComputeVariance(Matrix<>,Int32)` | Computes the variance of a single column. |
| `CovarianceToCorrelation(Matrix<>)` | Converts a covariance matrix to a correlation matrix. |
| `DiscoverStructure(Matrix<>,String[])` |  |
| `DiscoverStructure(Matrix<>,Vector<>,String[])` |  |
| `DiscoverStructureCore(Matrix<>)` | Core implementation that each algorithm must provide. |
| `GenerateDefaultNames(Int32)` | Generates default variable names like "X0", "X1", "X2", etc. |
| `GetColumn(Matrix<>,Int32)` | Extract a column from a matrix as a Vector for accelerated operations. |
| `GetRow(Matrix<>,Int32)` | Extract a row from a matrix as a Vector for accelerated operations. |
| `InvertSmallMatrix(Matrix<>)` | Inverts a small matrix using Gauss-Jordan elimination. |
| `MatMul(Matrix<>,Matrix<>)` | Accelerated matrix-matrix multiplication: C = A * B using Engine when available. |
| `MatVecMul(Matrix<>,Vector<>)` | Accelerated matrix-vector multiplication: result = M * v. |
| `MatrixExponentialTaylor(Matrix<>,Int32,Int32)` | Computes the matrix exponential via Taylor series: exp(M) = sum_{k=0}^{terms} M^k / k! |
| `NormalQuantile(Double)` | Standard normal quantile (inverse CDF) approximation using rational approximation. |
| `RowDotProduct(Matrix<>,Int32,Matrix<>,Int32,Int32)` | Accelerated dot product between a row of matrix A and a row of matrix B. |
| `SolveSmallSystem(Double[0:,0:],Double[],Int32)` | Solves a small linear system Ax = b using Gaussian elimination with partial pivoting. |
| `TestConditionalIndependence(Matrix<>,Int32,Int32,Int32[],Int32,Double)` | Tests conditional independence between variables i and j given a conditioning set. |
| `ThresholdMatrix(Matrix<>,Double)` | Applies a threshold to the adjacency matrix, zeroing out entries below the threshold. |
| `ValidateInput(Matrix<>)` | Validates that the input data matrix is suitable for causal discovery. |
| `VectorAdd(Vector<>,Vector<>)` | Accelerated element-wise addition of two vectors. |
| `VectorScale(Vector<>,)` | Accelerated scalar multiplication: a * scalar. |
| `VectorSubtract(Vector<>,Vector<>)` | Accelerated element-wise subtraction: a - b. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations helper for generic math on type T. |

