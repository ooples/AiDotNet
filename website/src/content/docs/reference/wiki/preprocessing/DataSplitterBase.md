---
title: "DataSplitterBase<T>"
description: "Base class for all data splitters providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Preprocessing.DataPreparation`

Base class for all data splitters providing common functionality.

## For Beginners

This base class provides shared utilities that all data splitters need:

- Shuffling indices randomly
- Selecting rows from matrices
- Validating inputs
- Working with both Matrix and Tensor data

## How It Works

When creating a custom splitter, inherit from this class to get these utilities for free.
You only need to implement the specific splitting logic for your algorithm.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DataSplitterBase(Boolean,Int32)` | Initializes a new instance of the DataSplitterBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `NumSplits` |  |
| `RequiresLabels` |  |
| `SupportsValidation` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildResult(Matrix<>,Vector<>,Int32[],Int32[],Int32[],Nullable<Int32>,Nullable<Int32>,Nullable<Int32>,Nullable<Int32>)` | Builds a DataSplitResult from computed indices. |
| `BuildTensorResult(Tensor<>,Tensor<>,Int32[],Int32[],Int32[],Nullable<Int32>,Nullable<Int32>,Nullable<Int32>,Nullable<Int32>)` | Builds a TensorSplitResult from computed indices. |
| `ComputeSplitSizes(Int32,Double,Double)` | Computes sizes for train/validation/test splits based on ratios. |
| `CopySample(Tensor<>,Tensor<>,Int32,Int32)` | Copies a single sample from source tensor to destination tensor. |
| `GetIndices(Int32)` | Creates an array of indices from 0 to count-1. |
| `GetShuffledIndices(Int32)` | Gets indices, optionally shuffled. |
| `GetSplits(Matrix<>,Vector<>)` |  |
| `GetTensorSplits(Tensor<>,Tensor<>)` |  |
| `GetUniqueLabels(Vector<>)` | Gets unique class labels from a target vector. |
| `GroupByLabel(Vector<>)` | Groups sample indices by their class label. |
| `SelectElements(Vector<>,Int32[])` | Selects specific elements from a vector. |
| `SelectRows(Matrix<>,Int32[])` | Selects specific rows from a matrix. |
| `SelectSamples(Tensor<>,Int32[])` | Selects specific samples from a tensor. |
| `ShuffleIndices(Int32[])` | Shuffles an array of indices in place using Fisher-Yates algorithm. |
| `Split(Matrix<>,Vector<>)` |  |
| `SplitIndicesOnly(Int32,Vector<>)` | Helper method to get split indices without building full results. |
| `SplitTensor(Tensor<>,Tensor<>)` |  |
| `ValidateInputs(Matrix<>,Vector<>)` | Validates that the input matrix and optional vector are compatible. |
| `ValidateTensorInputs(Tensor<>,Tensor<>)` | Validates tensor inputs. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for generic type T. |
| `_random` | Random number generator initialized with the seed. |
| `_randomSeed` | The random seed for reproducible splits. |
| `_shuffle` | Whether to shuffle data before splitting. |

