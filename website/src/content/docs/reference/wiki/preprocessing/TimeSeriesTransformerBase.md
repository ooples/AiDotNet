---
title: "TimeSeriesTransformerBase<T>"
description: "Abstract base class for all time series feature extractors providing shared functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Preprocessing.TimeSeries`

Abstract base class for all time series feature extractors providing shared functionality.

## For Beginners

This base class handles the common tasks that all time series
transformers need, so each specific transformer can focus on its unique calculations.

Features provided:

- NaN padding at the start where we don't have enough history
- Fast parallel processing for large datasets
- Automatic detection of optimal window sizes
- Consistent naming of output features

## How It Works

This class provides common functionality for time series transformers including:

- Window validation and edge handling
- Parallel processing for large datasets
- Auto-detection of optimal window sizes
- Feature naming conventions

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeSeriesTransformerBase(TimeSeriesFeatureOptions)` | Creates a new instance of the time series transformer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDetectEnabled` |  |
| `ColumnIndices` |  |
| `FeatureNames` |  |
| `InputFeatureCount` |  |
| `IsFitted` |  |
| `NumOps` | Gets the numeric operations helper for type T. |
| `OutputFeatureCount` |  |
| `SupportsIncrementalTransform` |  |
| `SupportsInverseTransform` |  |
| `WindowSizes` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyForwardFill(Tensor<>,Int32)` | Applies forward fill to output tensor for edge values. |
| `ApplyHannWindow(Double[],Int32)` | Applies a Hann window to reduce spectral leakage. |
| `ComputeAutocorrelation(Tensor<>,Int32)` | Computes the autocorrelation function for the time series. |
| `ComputeFFT(Double[])` | Computes the FFT using the Cooley-Tukey radix-2 algorithm. |
| `ComputeFeaturesForWindow(Double[],)` | Computes features for a window of data. |
| `ComputeIncrementalFeatures(IncrementalState<>,[])` | Computes features for a single data point using the incremental state. |
| `DetectOptimalWindowSizes(Tensor<>)` |  |
| `DetectUsingAutocorrelation(Tensor<>)` | Detects optimal window sizes using autocorrelation function analysis. |
| `DetectUsingGridSearch(Tensor<>)` | Detects optimal window sizes using grid search with cross-validation. |
| `DetectUsingHeuristics(Tensor<>)` | Detects optimal window sizes using simple heuristic rules. |
| `DetectUsingSpectralAnalysis(Tensor<>)` | Detects optimal window sizes using spectral analysis (FFT). |
| `EnsureFitted` | Ensures the transformer has been fitted. |
| `EvaluateWindowSize(Tensor<>,Int32,Int32,Int32)` | Evaluates how well a window size captures patterns in the data. |
| `ExportOptions` | Exports the options used to configure this transformer. |
| `ExportParameters` | Exports transformer-specific parameters. |
| `ExportState` | Exports the transformer's state for serialization. |
| `ExtractIncrementalWindow(IncrementalState<>,Int32,Int32)` | Extracts a window of data from the incremental buffer. |
| `FindAutocorrelationPeaks([],Int32,Int32)` | Finds peaks in the autocorrelation function. |
| `FindSpectralPeaks(Double[],Int32,Int32)` | Finds peaks in the power spectrum. |
| `Fit(Tensor<>)` |  |
| `FitCore(Tensor<>)` | Performs the core fitting logic specific to this transformer. |
| `FitTransform(Tensor<>)` |  |
| `GenerateCandidateWindowSizes(Int32)` | Generates candidate window sizes for grid search. |
| `GenerateDefaultInputNames(Int32)` | Generates default input feature names. |
| `GenerateFeatureNames` | Generates the feature names for this transformer's outputs. |
| `GetEffectiveWindowSize(Int32,Int32)` | Gets the effective window size for partial edge handling. |
| `GetFeatureNamesOut(String[])` |  |
| `GetIncrementalState` |  |
| `GetInputFeatureNames` | Gets the input feature names (learned during fitting or from options). |
| `GetMaxWindowSize` | Gets the maximum window size from configured windows. |
| `GetNaN` | Gets the NaN value for type T. |
| `GetOperationNames` | Gets the list of statistics or operations this transformer computes. |
| `GetOutputStartIndex` | Gets the starting time step index for output (used with Truncate mode). |
| `GetOutputTimeSteps(Int32)` | Determines the output row count based on edge handling mode. |
| `GetSeparator` | Gets the feature name separator from options. |
| `GetTimeSteps(Tensor<>)` | Gets the number of time steps in the data. |
| `GetValidationErrors(Tensor<>)` |  |
| `GetValue(Tensor<>,Int32,Int32)` | Gets a value from the tensor at the specified time step and feature. |
| `ImportOptions(Dictionary<String,Object>)` | Imports options from a dictionary. |
| `ImportParameters(Dictionary<String,Object>)` | Imports transformer-specific parameters. |
| `ImportState(TransformerState<>)` | Imports a previously exported state to restore the transformer. |
| `InitializeExtendedState(Tensor<>)` | Initializes any extended state needed for incremental processing. |
| `InitializeIncremental(Tensor<>)` |  |
| `InverseTransform(Tensor<>)` |  |
| `InverseTransformCore(Tensor<>)` | Performs the inverse transformation if supported. |
| `IsEdgeRegion(Int32,Int32)` | Determines if a time step is in the edge region (incomplete window). |
| `IsNaN()` | Checks if a value is NaN. |
| `NextPowerOfTwo(Int32)` | Returns the next power of 2 greater than or equal to n. |
| `SetIncrementalState(IncrementalState<>)` |  |
| `SetValue(Tensor<>,Int32,Int32,)` | Sets a value in the tensor at the specified time step and feature. |
| `ShouldComputePartialWindows` | Checks if the current edge handling mode should compute partial windows. |
| `Transform(Tensor<>)` |  |
| `TransformCore(Tensor<>)` | Performs the core transformation logic specific to this transformer. |
| `TransformIncremental([])` |  |
| `TransformParallel(Tensor<>)` | Transforms data using parallel processing for improved performance. |
| `ValidateInput(Tensor<>)` |  |
| `ValidateInputForFitting(Tensor<>)` | Validates input data for fitting. |
| `ValidateInputForTransform(Tensor<>)` | Validates input data for transformation. |

## Fields

| Field | Summary |
|:-----|:--------|
| `Options` | The configuration options for this transformer. |
| `_featureNames` | The generated feature names. |
| `_incrementalState` | The incremental state for streaming processing. |
| `_inputFeatureCount` | The number of input features learned during fitting. |
| `_inputFeatureNames` | The input feature names learned during fitting. |
| `_outputFeatureCount` | The number of output features that will be generated. |
| `_windowSizes` | The computed window sizes after auto-detection (if enabled). |

