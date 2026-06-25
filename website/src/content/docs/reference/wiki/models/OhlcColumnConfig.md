---
title: "OhlcColumnConfig"
description: "Configuration for OHLC (Open, High, Low, Close) column indices."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration for OHLC (Open, High, Low, Close) column indices.

## For Beginners

OHLC is standard financial data format:

- Open: First price when trading started
- High: Highest price during the period
- Low: Lowest price during the period
- Close: Last price when trading ended

These four values together tell the full story of price movement during each period.
Volatility estimators like Parkinson and Garman-Klass use this information to more
accurately measure how volatile (risky) an asset is.

## How It Works

This class specifies which columns in your data contain OHLC prices, enabling proper
calculation of volatility measures like Parkinson and Garman-Klass that require
high/low/open/close values.

## Properties

| Property | Summary |
|:-----|:--------|
| `CloseIndex` | Gets or sets the column index for Close prices. |
| `HasHighLow` | Checks if this configuration has valid High/Low indices for Parkinson volatility. |
| `HasOhlc` | Checks if this configuration has valid OHLC indices for Garman-Klass volatility. |
| `HighIndex` | Gets or sets the column index for High prices. |
| `LowIndex` | Gets or sets the column index for Low prices. |
| `OpenIndex` | Gets or sets the column index for Open prices. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateHlc(Int32,Int32,Int32)` | Creates a configuration for data with only High, Low, Close (no Open). |
| `CreateStandard` | Creates a standard OHLC configuration assuming columns are in order: Open, High, Low, Close. |

