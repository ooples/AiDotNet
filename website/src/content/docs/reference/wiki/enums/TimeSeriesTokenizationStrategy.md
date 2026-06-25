---
title: "TimeSeriesTokenizationStrategy"
description: "Tokenization strategies for time series foundation models."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Tokenization strategies for time series foundation models.

## For Beginners

Different foundation models use different ways to convert raw time
series data into "tokens" that the model can process:

- **Patching:** Splits the series into fixed-size chunks (most common)
- **Quantization:** Converts continuous values to discrete vocabulary tokens
- **AdaptivePatching:** Variable-size patches based on local complexity
- **LagFeatures:** Uses historical lags as features instead of raw values
- **RawSequence:** No tokenization — feeds raw values directly

## How It Works

**Reference:** See Chronos (ICML 2024) for quantization, PatchTST (ICLR 2023) for
patching, Kairos (2025) for adaptive patching, and Lag-Llama (2023) for lag features.

## Fields

| Field | Summary |
|:-----|:--------|
| `AdaptivePatching` | Variable-size patches based on local information density. |
| `LagFeatures` | Historical lag values as input features. |
| `Patching` | Non-overlapping or overlapping fixed-size patches. |
| `Quantization` | Discrete vocabulary quantization via uniform or learned binning. |
| `RawSequence` | Raw numerical sequence without tokenization. |

