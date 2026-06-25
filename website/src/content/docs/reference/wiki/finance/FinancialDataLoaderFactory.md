---
title: "FinancialDataLoaderFactory"
description: "Factory helpers for creating financial data loaders."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Finance.Data`

Factory helpers for creating financial data loaders.

## For Beginners

Use these helpers when you have OHLCV price data and want
a ready-to-use FinancialDataLoader for training or evaluation.

## How It Works

This factory keeps common loader configurations in one place so callers can
create forecasting loaders without repeating boilerplate setup.

## Methods

| Method | Summary |
|:-----|:--------|
| `FromProvider(MarketDataProvider<>,Int32,Int32,Boolean,Boolean,Boolean,Boolean,FinancialPreprocessor<>,Int32)` | Creates a financial data loader from a market data provider. |
| `FromSeries(IReadOnlyList<MarketDataPoint<>>,Int32,Int32,Boolean,Boolean,Boolean,Boolean,FinancialPreprocessor<>,Int32)` | Creates a financial data loader from a list of market data points. |

