---
title: "AdditiveStructureKernel<T>"
description: "Additive Structure Kernel that decomposes the function into additive components."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Additive Structure Kernel that decomposes the function into additive components.

## For Beginners

The Additive Structure Kernel assumes that the target function
can be decomposed into a sum of lower-dimensional components:

f(x) = f₁(x₁) + f₂(x₂) + ... + fₘ(x_groupₘ)

Where each fᵢ depends only on a subset of input features (a "group").

This is useful because:

1. **Interpretability**: You can understand how each feature (or group) contributes
2. **Efficiency**: Additive structure can reduce computational complexity
3. **Generalization**: Simpler models often generalize better

Example: Predicting house prices might be additive:
price = location_effect(lat, lon) + size_effect(sqft) + age_effect(year_built)

The kernel for additive structure is:
k(x, x') = Σᵢ kᵢ(x_groupᵢ, x'_groupᵢ)

Each component can have its own kernel (e.g., RBF for smooth components,
periodic for cyclical features).

## How It Works

Applications:

- Feature importance analysis
- Structured time series (trend + seasonality)
- Scientific modeling with known additive structure
- High-dimensional problems where full interactions are intractable

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdditiveStructureKernel(IKernelFunction<>[],Int32[][],Double[])` | Initializes an Additive Structure Kernel with specified components. |
| `AdditiveStructureKernel(Int32,IKernelFunction<>,Double[])` | Initializes an Additive Structure Kernel with one kernel per feature (fully additive). |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumComponents` | Gets the number of additive components. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the additive kernel value. |
| `CalculateComponent(Vector<>,Vector<>,Int32)` | Calculates the kernel value for a specific component only. |
| `ExtractFeatures(Vector<>,Int32[])` | Extracts specified features from a vector. |
| `GetComponentImportances(Vector<>)` | Computes the importance of each component based on its kernel value at the origin. |
| `GetFeatureGroup(Int32)` | Gets the feature group for a component. |
| `GetWeight(Int32)` | Gets the weight for a component. |
| `WithRBF(Int32,Double)` | Creates a simple additive kernel with RBF components. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_componentKernels` | The component kernels. |
| `_featureGroups` | The feature indices for each component. |
| `_numComponents` | Number of components. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_weights` | The weights for each component. |

