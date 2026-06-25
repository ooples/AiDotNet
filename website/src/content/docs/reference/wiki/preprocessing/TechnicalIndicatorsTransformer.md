---
title: "TechnicalIndicatorsTransformer<T>"
description: "Computes technical analysis indicators for time series data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TimeSeries`

Computes technical analysis indicators for time series data.

## For Beginners

Technical indicators are mathematical formulas applied to price/volume data
to help predict future price movements. They fall into several categories:

- **Moving Averages**: Smooth out price data to identify trends (SMA, EMA, WMA, DEMA, TEMA)
- **Momentum Indicators**: Measure the speed of price changes (RSI, MACD, Stochastic, CCI)
- **Volatility Indicators**: Measure how much prices are fluctuating (Bollinger Bands, ATR)
- **Volume Indicators**: Confirm trends using trading volume (OBV)

Traders use these indicators to:

- Identify when to buy or sell
- Confirm trend strength
- Spot potential reversals
- Set stop-loss levels

## How It Works

This transformer calculates industry-standard technical indicators commonly used in
financial analysis and algorithmic trading, including moving averages, momentum indicators,
volatility bands, and volume-based indicators.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TechnicalIndicatorsTransformer(TimeSeriesFeatureOptions)` | Creates a new technical indicators transformer with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsIncrementalTransform` | Gets whether this transformer supports incremental transformation. |
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeADX(Double[],Double[],Double[],Int32)` | Computes Average Directional Index (ADX) using actual OHLC data. |
| `ComputeATR(Double[],Double[],Double[],Int32)` | Computes Average True Range (ATR) using actual OHLC data. |
| `ComputeATRApproximation(Double[],Int32)` | Computes ATR approximation when OHLC data is not available. |
| `ComputeBollingerBands(Double[],Int32,Double)` | Computes Bollinger Bands (middle band = SMA, upper/lower = SMA ± stdDev*multiplier). |
| `ComputeCCI(Double[],Double[],Double[],Int32)` | Computes Commodity Channel Index using actual OHLC data. |
| `ComputeCCIApproximation(Double[],Int32)` | Computes CCI approximation when OHLC data is not available. |
| `ComputeDEMA(Double[],Int32)` | Computes Double Exponential Moving Average (DEMA = 2*EMA - EMA(EMA)). |
| `ComputeEMA(Double[],Int32)` | Computes Exponential Moving Average. |
| `ComputeMACD(Double[],Int32,Int32,Int32)` | Computes MACD (Moving Average Convergence Divergence). |
| `ComputeOBVApproximation(Double[])` | Computes On-Balance Volume approximation using price as proxy for volume. |
| `ComputeRSI(Double[],Int32)` | Computes Relative Strength Index (RSI). |
| `ComputeSMA(Double[],Int32)` | Computes Simple Moving Average. |
| `ComputeStochastic(Double[],Double[],Double[],Int32,Int32)` | Computes Stochastic Oscillator using actual OHLC data. |
| `ComputeStochasticApproximation(Double[],Int32,Int32)` | Computes Stochastic Oscillator approximation when OHLC data is not available. |
| `ComputeTEMA(Double[],Int32)` | Computes Triple Exponential Moving Average (TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))). |
| `ComputeWMA(Double[],Int32)` | Computes Weighted Moving Average. |
| `ComputeWilliamsR(Double[],Double[],Double[],Int32)` | Computes Williams %R using actual OHLC data. |
| `ComputeWilliamsRApproximation(Double[],Int32)` | Computes Williams %R approximation when OHLC data is not available. |
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
| `_adxPeriod` | ADX period. |
| `_bollingerStdDev` | Bollinger Band standard deviation multiplier. |
| `_cciPeriod` | CCI period. |
| `_enabledIndicators` | The enabled technical indicators. |
| `_longPeriod` | Long period for EMA/MACD calculations. |
| `_ohlcConfig` | OHLC column configuration. |
| `_operationNames` | Cached operation names. |
| `_rsiPeriod` | RSI period. |
| `_shortPeriod` | Short period for EMA/MACD calculations. |
| `_signalPeriod` | Signal period for MACD signal line. |
| `_stochasticDPeriod` | Stochastic D period. |
| `_stochasticKPeriod` | Stochastic K period. |

