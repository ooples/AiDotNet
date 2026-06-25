---
title: "SyntheticTabularGeneratorBase<T>"
description: "Abstract base class for synthetic tabular data generators, providing common infrastructure for fitting models on real data and generating synthetic rows."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NeuralNetworks.SyntheticData`

Abstract base class for synthetic tabular data generators, providing common infrastructure
for fitting models on real data and generating synthetic rows.

## For Beginners

This is the shared foundation that CTGAN, TVAE, and TabDDPM build on.
It handles the "bookkeeping" that every generator needs:

- Remembering column descriptions (which columns are numbers vs categories)
- Managing the random number generator (for reproducibility)
- Tracking whether the model has been trained yet
- Computing basic statistics (min, max, mean, std) for each column

Specific generators override `Int32)` and `Vector{`
to implement their unique training and generation algorithms.

## How It Works

This base class handles the lifecycle shared by all tabular generators:
column metadata management, random number generation, fitted-state tracking,
and the public Fit/Generate API that delegates to subclass implementations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SyntheticTabularGeneratorBase(Nullable<Int32>)` | Initializes a new instance of the base class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClipGradientNorm(Tensor<>,Double)` | Clips a gradient tensor to a maximum L2 norm, preventing exploding gradients. |
| `ClipGradientNorm(Vector<>,Double)` | Clips a gradient vector to a maximum L2 norm, preventing exploding gradients. |
| `ComputeColumnStatistics(Matrix<>,Int32,ColumnMetadata)` | Computes min, max, mean, and standard deviation for a numerical column. |
| `CreateStandardNormalVector(Int32)` | Creates a vector filled with standard normal random values. |
| `FillStandardNormal(Vector<>)` | Fills a vector with standard normal random values. |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` |  |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `FitInternal(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` | Subclass-specific training implementation. |
| `Generate(Int32,Vector<>,Vector<>)` |  |
| `GenerateInternal(Int32,Vector<>,Vector<>)` | Subclass-specific generation implementation. |
| `HasNaN(Tensor<>)` | Checks whether a tensor contains any NaN or Infinity values. |
| `HasNaN(Vector<>)` | Checks a vector for NaN/Infinity values, returning true if any are found. |
| `IsFiniteLoss(Double)` | Checks whether a loss value is valid (not NaN or Infinity). |
| `PrepareColumns(Matrix<>,IReadOnlyList<ColumnMetadata>)` | Clones column metadata, assigns indices, and computes statistics from data. |
| `SafeGradient(Tensor<>,Double)` | Applies NaN sanitization and gradient norm clipping in a single operation. |
| `SafeGradient(Vector<>,Double)` | Applies NaN sanitization and gradient norm clipping to a vector in a single operation. |
| `SampleStandardNormal` | Samples a standard normal random value as type T. |
| `SanitizeTensor(Tensor<>)` | Replaces NaN and Infinity values in a tensor with zero, in-place. |
| `SanitizeTensor(Vector<>)` | Replaces NaN and Infinity values in a vector with zero, in-place. |
| `ValidateFitInputs(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` | Validates inputs to the Fit method. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides numeric operations for the specific type T. |
| `Random` | Random number generator for stochastic operations. |
| `_columns` | The stored column metadata after fitting. |

