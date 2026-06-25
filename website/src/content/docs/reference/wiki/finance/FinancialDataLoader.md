---
title: "FinancialDataLoader<T>"
description: "Data loader for financial time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Data`

Data loader for financial time series forecasting.

## For Beginners

This loader creates training samples like:

- Input: the last N time steps (lookback window)
- Output: the next M time steps (prediction horizon)

It handles batching, shuffling, and splitting into train/val/test.

## How It Works

FinancialDataLoader turns a list of OHLCV points into windowed tensors
suitable for training forecasting models. It implements the standard
InputOutputDataLoader API so it works with AiDotNet's training pipeline.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinancialDataLoader(IReadOnlyList<MarketDataPoint<>>,Int32,Int32,Boolean,Boolean,Boolean,Boolean,FinancialPreprocessor<>,Int32)` | Creates a new financial data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `FeatureCount` |  |
| `Name` |  |
| `OutputDimension` |  |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateSubsetLoader(Int32[])` | Creates a subset loader for a list of indices. |
| `ExtractBatch(Int32[])` |  |
| `ExtractTensorSubset(Tensor<>,Int32[])` | Extracts a subset of samples from a tensor. |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

