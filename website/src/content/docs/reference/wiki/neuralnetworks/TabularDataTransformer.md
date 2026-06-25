---
title: "TabularDataTransformer<T>"
description: "Transforms tabular data using Variational Gaussian Mixture (VGM) mode-specific normalization for continuous columns and one-hot encoding for categorical columns."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

Transforms tabular data using Variational Gaussian Mixture (VGM) mode-specific normalization
for continuous columns and one-hot encoding for categorical columns. Used by CTGAN and TVAE.

## For Beginners

Real-world data often has columns with very different distributions.
A "price" column might have values clustered around $10 and $100 (two modes), while an
"age" column might be normally distributed. Simple min-max or z-score normalization doesn't
handle multi-modal (multiple-peak) distributions well.

VGM normalization solves this by:

1. Fitting a Gaussian mixture (like fitting multiple bell curves) to each continuous column
2. For each value, finding which bell curve (mode) it belongs to
3. Normalizing relative to that mode (so both "cheap" and "expensive" items get reasonable values)
4. Adding a one-hot indicator showing which mode was chosen

This helps the generator learn multi-modal distributions much more effectively.

## How It Works

The transformation follows the CTGAN paper (Xu et al., NeurIPS 2019):

- **Continuous columns**: A Gaussian mixture model (GMM) is fitted per column.

Each value is then represented as (normalized_value, one-hot_mode_indicator),
where the normalized value is relative to the selected mode's mean and std.

- **Categorical columns**: One-hot encoded into binary indicator vectors.
- **Inverse transform**: Reconstructs original-scale values from the transformed representation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabularDataTransformer(Int32,Random)` | Initializes a new `TabularDataTransformer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` | Gets the column metadata this transformer was fitted on. |
| `IsFitted` | Gets whether this transformer has been fitted. |
| `TransformedWidth` | Gets the width of the transformed data representation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(BinaryReader)` | Restores transformer state previously written by `BinaryWriter)`. |
| `Digamma(Double)` | Digamma function ψ(x) = d/dx ln Γ(x), used by the variational GMM E-step. |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>)` | Fits the transformer to the data by learning VGM parameters for continuous columns and category mappings for categorical columns. |
| `GetTransformInfo(Int32)` | Gets the transform info for a specific original column index. |
| `InverseTransform(Matrix<>)` | Inverse-transforms generated data back to the original column space. |
| `SampleMode(TabularDataTransformer<>.VGMColumnInfo,Double)` | Samples a mode index for `value` proportional to the component responsibilities ρ_m ∝ w_m · N(value; μ_m, σ_m), matching the paper's probabilistic mode assignment. |
| `Serialize(BinaryWriter)` | Serializes the fitted transformer state (column metadata, per-column transform layout, and the fitted VGM / categorical parameters) so a saved or cloned generator can inverse-transform generated samples back to the original column space wit… |
| `Transform(Matrix<>)` | Transforms the raw data matrix into the VGM-normalized + one-hot representation. |

