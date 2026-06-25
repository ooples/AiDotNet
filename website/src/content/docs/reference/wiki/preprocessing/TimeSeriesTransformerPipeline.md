---
title: "TimeSeriesTransformerPipeline<T>"
description: "A pipeline that chains multiple time series transformers together."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TimeSeries`

A pipeline that chains multiple time series transformers together.

## For Beginners

Think of this like an assembly line in a factory:

Raw Material -> [Machine 1] -> [Machine 2] -> [Machine 3] -> Final Product

Similarly, your data flows through each transformer in order:

Raw Data -> [Lag Features] -> [Rolling Stats] -> [Technical Indicators] -> Enhanced Features

This makes it easy to build complex feature engineering workflows by combining
simple, focused transformers.

## How It Works

This class allows you to compose multiple transformers into a single pipeline,
applying them in sequence. Each transformer's output becomes the next transformer's input.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeSeriesTransformerPipeline` | Creates a new pipeline with default settings. |
| `TimeSeriesTransformerPipeline(Boolean,PipelineMode)` | Creates a new pipeline with the specified settings. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDetectEnabled` |  |
| `ColumnIndices` |  |
| `FeatureNames` |  |
| `InputFeatureCount` |  |
| `IsFitted` |  |
| `Item(Int32)` | Gets the transformer at the specified index. |
| `OutputFeatureCount` |  |
| `SupportsIncrementalTransform` |  |
| `SupportsInverseTransform` |  |
| `TransformerCount` | Gets the number of transformers in the pipeline. |
| `WindowSizes` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddTransformer(ITimeSeriesFeatureExtractor<>)` | Adds a transformer to the end of the pipeline. |
| `AddTransformers(IEnumerable<ITimeSeriesFeatureExtractor<>>)` | Adds multiple transformers to the pipeline. |
| `AddTransformers(ITimeSeriesFeatureExtractor<>[])` | Adds multiple transformers to the pipeline. |
| `Clone` | Creates a deep copy of this pipeline (transformers are not copied, only the pipeline structure). |
| `ConcatenateFeatures(List<Tensor<>>,Int32)` | Concatenates multiple output tensors along the feature dimension. |
| `DetectOptimalWindowSizes(Tensor<>)` |  |
| `EnsureFitted` | Ensures the pipeline has been fitted. |
| `Fit(Tensor<>)` |  |
| `FitParallel(Tensor<>)` | Fits transformers in parallel mode (all receive original data). |
| `FitSequential(Tensor<>)` | Fits transformers in sequential mode. |
| `FitTransform(Tensor<>)` |  |
| `GetFeatureNamesOut(String[])` |  |
| `GetIncrementalState` |  |
| `GetSummary` | Gets a summary of the pipeline configuration. |
| `GetValidationErrors(Tensor<>)` |  |
| `InitializeIncremental(Tensor<>)` |  |
| `InverseTransform(Tensor<>)` |  |
| `SetIncrementalState(IncrementalState<>)` |  |
| `Transform(Tensor<>)` |  |
| `TransformIncremental([])` |  |
| `TransformParallel(Tensor<>)` | Transforms data in parallel mode (concatenates all transformer outputs). |
| `TransformSequential(Tensor<>)` | Transforms data in sequential mode. |
| `ValidateInput(Tensor<>)` |  |
| `ValidateInputForTransform(Tensor<>)` | Validates input data for transformation. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_featureNames` | The combined feature names from all transformers. |
| `_includeOriginalFeatures` | Whether to concatenate original features with transformed features. |
| `_inputFeatureCount` | The number of features expected in the input data. |
| `_numOps` | Gets the numeric operations helper for type T. |
| `_outputFeatureCount` | The total number of features produced by the pipeline. |
| `_pipelineMode` | Whether each transformer receives only the original input (parallel mode) or the output of the previous transformer (sequential mode). |
| `_transformers` | The ordered list of transformers in this pipeline. |
| `_windowSizes` | The window sizes used by the pipeline (union of all transformer windows). |

