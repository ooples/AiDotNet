---
title: "RollingVolatilityTransformer<T>"
description: "Computes rolling volatility and return-based features for financial time series."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TimeSeries`

Computes rolling volatility and return-based features for financial time series.

## For Beginners

Volatility measures how much a price moves up and down over time.

High volatility means:

- Prices change a lot day-to-day
- Higher risk but also higher potential returns
- Common during market uncertainty

Low volatility means:

- Prices are relatively stable
- Lower risk but also lower potential returns
- Common in stable market conditions

This transformer creates features like:

- Simple returns: (Today's price - Yesterday's price) / Yesterday's price
- Log returns: ln(Today's price / Yesterday's price) - preferred for statistical analysis
- Realized volatility: Standard deviation of returns over a window
- Momentum: How much price has changed over a period

## How It Works

This transformer calculates volatility measures commonly used in quantitative finance,
including realized volatility, Parkinson, and Garman-Klass estimators.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RollingVolatilityTransformer(TimeSeriesFeatureOptions)` | Creates a new rolling volatility transformer with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeEwmaVolatility(Double[],Double)` | Computes EWMA (Exponentially Weighted Moving Average) volatility. |
| `ComputeGarchVolatility(Double[],Double,Double,Double)` | Computes GARCH(1,1) volatility estimate. |
| `ComputeGarmanKlassApproximation(Double[])` | Computes Garman-Klass volatility approximation when OHLC data is not available. |
| `ComputeGarmanKlassFromOhlc(Double[],Double[],Double[],Double[])` | Computes Garman-Klass volatility from actual OHLC data. |
| `ComputeGarmanKlassVolatility(Double[],Double[],Double[],Double[],Double[],Int32)` | Computes Garman-Klass volatility estimator using OHLC data when available. |
| `ComputeIncrementalFeatures(IncrementalState<>,[])` | Computes volatility features incrementally from the circular buffer. |
| `ComputeLogReturns(Double[])` | Computes log returns: ln(P_t / P_{t-1}). |
| `ComputeMomentum(Double[],Int32)` | Computes price momentum (rate of change). |
| `ComputeParkinsonApproximation(Double[])` | Computes Parkinson volatility approximation when OHLC data is not available. |
| `ComputeParkinsonFromOhlc(Double[],Double[])` | Computes Parkinson volatility from actual OHLC high/low data. |
| `ComputeParkinsonVolatility(Double[],Double[],Double[],Int32)` | Computes Parkinson volatility estimator using OHLC data when available. |
| `ComputeRogersSatchellVariance(Double[],Double[],Double[],Double[],Int32)` | Computes Rogers-Satchell variance component. |
| `ComputeRogersSatchellVolatility(Double[],Double[],Double[],Double[],Double[],Int32)` | Computes Rogers-Satchell volatility estimator. |
| `ComputeSimpleReturns(Double[])` | Computes simple returns: (P_t - P_{t-1}) / P_{t-1}. |
| `ComputeStdDev(Double[])` | Computes sample standard deviation. |
| `ComputeVolatilityFeaturesIncremental(Double[],[],Int32)` | Computes volatility features for a window of data incrementally. |
| `ComputeYangZhangVolatility(Double[],Double[],Double[],Double[],Double[],Int32)` | Computes Yang-Zhang volatility estimator using OHLC data. |
| `ExportParameters` | Exports transformer-specific parameters for serialization. |
| `FitCore(Tensor<>)` |  |
| `GenerateFeatureNames` |  |
| `GetOperationNames` |  |
| `ImportParameters(Dictionary<String,Object>)` | Imports transformer-specific parameters for validation. |
| `TransformCore(Tensor<>)` |  |
| `TransformParallel(Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_annualizationFactor` | The annualization factor for volatility scaling. |
| `_calculateMomentum` | Whether to calculate momentum. |
| `_calculateReturns` | Whether to calculate returns. |
| `_enabledMeasures` | The enabled volatility measures. |
| `_ewmaLambda` | EWMA decay factor (lambda). |
| `_garchAlpha` | GARCH alpha (squared return coefficient). |
| `_garchBeta` | GARCH beta (lagged variance coefficient). |
| `_garchOmega` | GARCH omega (constant term). |
| `_ohlcConfig` | OHLC column configuration for proper volatility calculations. |
| `_operationNames` | Cached operation names. |

