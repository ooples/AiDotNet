---
title: "CORALDomainAdapter<T>"
description: "Implements domain adaptation using CORrelation ALignment (CORAL)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TransferLearning.DomainAdaptation`

Implements domain adaptation using CORrelation ALignment (CORAL).

## For Beginners

CORAL (CORrelation ALignment) aligns the second-order statistics
(covariances) of source and target domains. Think of it as making sure the "spread" and
"correlation patterns" of features are similar in both domains.

## How It Works

Imagine you have two datasets: one where features vary a lot, and another where they
vary less. CORAL adjusts the data so both have similar variability patterns, making
transfer learning more effective.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CORALDomainAdapter` | Initializes a new instance of the CORALDomainAdapter class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationMethod` | Gets the name of the adaptation method. |
| `RequiresTraining` | Gets whether this adapter requires training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdaptSource(Matrix<>,Matrix<>)` | Adapts source data to match target distribution using CORAL. |
| `AdaptTarget(Matrix<>,Matrix<>)` | Adapts target data to match source distribution. |
| `CenterData(Matrix<>,Vector<>)` | Centers the data by subtracting the mean. |
| `ComputeCORALTransformation(Matrix<>,Matrix<>)` | Computes the CORAL transformation matrix. |
| `ComputeCovariance(Matrix<>)` | Computes the covariance matrix of the data. |
| `ComputeDomainDiscrepancy(Matrix<>,Matrix<>)` | Computes the domain discrepancy using Frobenius norm of covariance difference. |
| `ComputeFrobeniusNorm(Matrix<>)` | Computes the Frobenius norm of a matrix. |
| `ComputeMatrixInverseSqrt(Matrix<>)` | Computes the inverse square root of a matrix. |
| `ComputeMatrixSqrt(Matrix<>)` | Computes the matrix square root using eigendecomposition approximation. |
| `ComputeMean(Matrix<>)` | Computes the mean of each feature column. |
| `DecenterData(Matrix<>,Vector<>)` | Decenters the data by adding the mean. |
| `MatrixSubtract(Matrix<>,Matrix<>)` | Subtracts two matrices. |
| `Train(Matrix<>,Matrix<>)` | Trains the CORAL adapter by computing the transformation matrix. |

