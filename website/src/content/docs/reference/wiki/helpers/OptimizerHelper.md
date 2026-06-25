---
title: "OptimizerHelper<T, TInput, TOutput>"
description: "OptimizerHelper<T, TInput, TOutput> — Helpers & Utilities in AiDotNet.Helpers."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

_No summary documentation available yet._

## Methods

| Method | Summary |
|:-----|:--------|
| `CopyTensorSlice(Tensor<>,Tensor<>,Int32[],Int32[],Int32)` | Recursively copies tensor slices for higher-dimensional tensors. |
| `CreateDatasetResult(,ErrorStats<>,BasicStats<>,BasicStats<>,PredictionStats<>,,)` | Creates a result object containing evaluation metrics for a specific dataset (training, validation, or test). |
| `CreateOptimizationInputData(,,,,,)` | Creates a data container for optimization algorithms with training, validation, and test datasets. |
| `CreateOptimizationResult(IFullModel<,,>,,List<>,List<Vector<>>,OptimizationResult<,,>.DatasetResult,OptimizationResult<,,>.DatasetResult,OptimizationResult<,,>.DatasetResult,FitDetectorResult<>,Int32,List<Int32>)` | Creates a result object containing all information about an optimization process. |
| `GetFeatureIndex(Vector<>)` | Determines which feature is represented by a feature vector. |
| `SelectFeaturesMatrix(Matrix<>,List<Int32>)` | Selects specific features from a matrix of input data. |
| `SelectFeaturesTensor(Tensor<>,List<Int32>)` | Selects specific features from a tensor of input data. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Provides operations for the numeric type T (like addition, multiplication, etc.). |

