---
title: "KronosOptions<T>"
description: "Configuration options for Kronos (Foundation Model for the Language of Financial Markets)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Kronos (Foundation Model for the Language of Financial Markets).

## For Beginners

Kronos is purpose-built for financial markets:

**Candlestick-Native Architecture:**
Unlike general time series models that treat financial data as simple numbers,
Kronos understands candlestick (K-line) patterns directly. Each time step has
5 features: Open, High, Low, Close, Volume (OHLCV).

**Key Advantages:**

- Pretrained on 12B+ financial records from 45 exchanges
- Natively handles multi-feature candlestick data
- Decoder-only architecture (efficient autoregressive generation)

**When to Use:**

- Financial market forecasting with candlestick data
- When you need a model that understands OHLCV patterns

## How It Works

Kronos is a decoder-only foundation model pre-trained on 12B+ K-line (candlestick) records
across 45 global exchanges. It natively understands OHLCV (Open, High, Low, Close, Volume)
candlestick patterns for financial market forecasting.

**Reference:** "Kronos: A Foundation Model for the Language of Financial Markets", 2025.
https://arxiv.org/abs/2508.02739

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KronosOptions` | Initializes a new instance with default values. |
| `KronosOptions(KronosOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the number of historical candlestick steps used as input context. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the number of future candlestick steps to forecast. |
| `HiddenDimension` | Gets or sets the hidden dimension of the transformer layers. |
| `IntermediateSize` | Gets or sets the intermediate size for the feed-forward network. |
| `ModelSize` | Gets or sets the model size variant. |
| `NumCandlestickFeatures` | Gets or sets the number of candlestick features (OHLCV = 5). |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `PatchLength` | Gets or sets the patch length for input tokenization. |

