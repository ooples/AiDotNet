---
title: "MarketDataProvider<T>"
description: "Stores and serves market data for finance workflows."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Data`

Stores and serves market data for finance workflows.

## For Beginners

This is like a small database of price bars. You can
add data points, slice out date ranges, or convert the series into tensors
for model training.

## How It Works

MarketDataProvider is a lightweight in-memory store for OHLCV data.
It is intentionally simple so it can be used by loaders, preprocessors,
and trading environments without extra dependencies.

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of data points stored. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(MarketDataPoint<>)` | Adds a single market data point. |
| `AddRange(IEnumerable<MarketDataPoint<>>)` | Adds multiple market data points at once. |
| `Clear` | Clears all stored market data points. |
| `GetAll` | Returns all stored points as a read-only list. |
| `GetFeatureCount(Boolean)` | Computes the feature count for OHLCV tensors. |
| `GetRange(DateTime,DateTime)` | Returns points within a specific time range. |
| `GetWindow(Int32,Int32)` | Returns a fixed-size window of points by index. |
| `ToTensor(Boolean)` | Converts the stored OHLCV data into a tensor. |

