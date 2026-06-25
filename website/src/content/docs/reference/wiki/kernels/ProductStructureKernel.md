---
title: "ProductStructureKernel<T>"
description: "Product Structure Kernel for modeling multiplicative interactions between feature groups."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Product Structure Kernel for modeling multiplicative interactions between feature groups.

## For Beginners

The Product Structure Kernel assumes the underlying function has
a multiplicative structure: f(x) = f_1(x_G1) × f_2(x_G2) × ... × f_K(x_GK)

Where x_Gi is the subset of features in group i.

The kernel is: k(x, x') = k_1(x_G1, x'_G1) × k_2(x_G2, x'_G2) × ... × k_K(x_GK, x'_GK)

This is useful when:

- Features naturally group together (e.g., spatial × temporal)
- There's known multiplicative interaction structure
- You want to reduce computation via Kronecker structure

Example: Modeling sales as (location effects) × (time effects) × (product effects)

Compare to Additive Structure:

- Additive: f = f_1 + f_2 + ... (sum of independent effects)
- Product: f = f_1 × f_2 × ... (multiplicative interaction)

Product kernels can capture stronger interactions but are less interpretable.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProductStructureKernel(IKernelFunction<>[],Int32[][])` | Initializes a Product Structure Kernel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureGroups` | Gets the feature indices for each group. |
| `NumGroups` | Gets the number of feature groups. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the product kernel value. |
| `CholeskyDecomposition(Matrix<>)` | Cholesky decomposition. |
| `ComputeGroupKernelMatrices(Matrix<>[])` | Computes the kernel matrix with Kronecker structure (for gridded data). |
| `FullyFactorized(Int32,Double)` | Creates a Product Structure Kernel with automatic grouping (one feature per group). |
| `GetRow(Matrix<>,Int32)` | Extracts a row from a matrix. |
| `KroneckerMultiply(Matrix<>[],Vector<>)` | Performs efficient Kronecker matrix-vector product. |
| `LogDeterminant(Matrix<>[])` | Computes log-determinant using Kronecker structure. |
| `WithRBF(Int32[][],Double[])` | Creates a Product Structure Kernel with RBF for all groups. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_featureGroups` | Feature indices for each group. |
| `_groupKernels` | Kernels for each feature group. |
| `_numOps` | Operations for performing numeric calculations with type T. |

