---
title: "DifferencingTransformer<T>"
description: "Applies differencing and stationarity transformations to time series data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TimeSeries`

Applies differencing and stationarity transformations to time series data.

## For Beginners

Many forecasting methods assume your data doesn't have trends
or seasonal patterns. This transformer removes those patterns through differencing.

Common transforms include:

- **First Difference**: Change from previous value (removes linear trend)
- **Seasonal Difference**: Change from same time in previous season (removes seasonality)
- **Detrending**: Subtract a fitted trend line
- **Decomposition**: Separate trend, seasonal, and residual components

After transformation, you can check stationarity using statistical tests like
the Augmented Dickey-Fuller test.

## How It Works

This transformer provides various methods to make time series stationary, which is
a requirement for many forecasting models (ARIMA, VAR, etc.). Stationary data has
constant statistical properties over time.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DifferencingTransformer(TimeSeriesFeatureOptions)` | Creates a new differencing transformer with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDetrended(Double[],Double[],Int32)` | Computes detrended series by subtracting fitted polynomial. |
| `ComputeDifference(Double[],Int32)` | Computes n-th order differencing. |
| `ComputeHodrickPrescott(Double[],Double)` | Computes Hodrick-Prescott filter decomposition. |
| `ComputeIncrementalFeatures(IncrementalState<>,[])` | Computes differencing features incrementally from the circular buffer. |
| `ComputeLogDifference(Double[])` | Computes log difference (log returns). |
| `ComputePercentChange(Double[])` | Computes percent change. |
| `ComputeSeasonalDifference(Double[],Int32)` | Computes seasonal differencing. |
| `ComputeStlDecomposition(Double[],Int32,Int32)` | Computes STL (Seasonal-Trend decomposition using LOESS) decomposition. |
| `CopyToOutput(Tensor<>,Double[],Int32)` | Copies a double array to the output tensor at the specified column. |
| `ExportParameters` | Exports transformer-specific parameters for serialization. |
| `ExtractSeries(Tensor<>,Int32,Int32)` | Extracts a single feature series from tensor. |
| `FitCore(Tensor<>)` |  |
| `FitPolynomial(Double[],Int32)` | Fits a polynomial to the series using least squares. |
| `GenerateFeatureNames` |  |
| `GetOperationNames` |  |
| `ImportParameters(Dictionary<String,Object>)` | Imports transformer-specific parameters for validation. |
| `InterpolateNaN(Double[])` | Interpolates NaN values using linear interpolation. |
| `SmoothSubseries(List<ValueTuple<Int32,Double>>)` | Smooths a subseries using weighted average. |
| `SmoothTrend(Double[],Int32)` | Smooths trend using moving average. |
| `SolveNormalEquations(Double[0:,0:],Double[],Int32,Int32)` | Solves normal equations using Cholesky decomposition. |
| `SolvePentadiagonal(Double[],Double[],Double[],Double[],Double[],Double[])` | Solves a pentadiagonal system Ax = r using Gaussian elimination. |
| `TransformCore(Tensor<>)` |  |
| `TransformParallel(Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_differencingOrder` | Differencing order. |
| `_enabledFeatures` | The enabled differencing features. |
| `_featureNames` | Cached feature names. |
| `_hpLambda` | Hodrick-Prescott filter lambda. |
| `_polynomialDegree` | Polynomial degree for detrending. |
| `_seasonalPeriod` | Seasonal differencing period. |
| `_stlIterations` | STL robust iterations. |
| `_stlPeriod` | STL seasonal period. |
| `_trendCoefficients` | Fitted trend coefficients for detrending (per feature). |

