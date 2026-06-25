---
title: "MarketDataPoint<T>"
description: "Represents a single market data point (OHLCV)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Data`

Represents a single market data point (OHLCV).

## For Beginners

Think of this as one row in a stock chart: it tells you
the price when the period opened, the highest and lowest prices, the closing
price, and how much volume traded during that period.

## How It Works

This class stores one time-stamped bar of market data, including open, high,
low, close, and volume values. It is the basic unit of financial time series
used by data loaders and trading environments.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MarketDataPoint(DateTime,,,,,)` | Creates a new market data point. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Close` | Gets the close price. |
| `High` | Gets the high price. |
| `Low` | Gets the low price. |
| `Open` | Gets the open price. |
| `Timestamp` | Gets the timestamp for the market data point. |
| `Volume` | Gets the traded volume. |

