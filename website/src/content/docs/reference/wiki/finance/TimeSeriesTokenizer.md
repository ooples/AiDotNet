---
title: "TimeSeriesTokenizer<T>"
description: "Standard tokenization pipeline for time series foundation models, supporting patching, quantization, and instance normalization strategies."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Tokenization`

Standard tokenization pipeline for time series foundation models, supporting patching,
quantization, and instance normalization strategies.

## For Beginners

Before feeding raw time series data into a foundation model,
we need to "tokenize" it — convert it into a format the model expects. Different models
use different tokenization strategies:

- **Patching** (PatchTST, Chronos-2, Moirai): Splits the series into fixed-size chunks
- **Quantization** (Chronos v1): Maps continuous values to discrete vocabulary tokens
- **Instance Normalization** (RevIN): Normalizes each series independently
- **Adaptive** (Kairos): Variable-size patches based on local information density

## How It Works

**Reference:** Standard tokenization approaches from Chronos (ICML 2024),
PatchTST (ICLR 2023), and Kairos (2025).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeSeriesTokenizer(TimeSeriesTokenizationStrategy,Int32,Nullable<Int32>,Int32)` | Creates a new tokenizer with the specified strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `PatchLength` | Gets the patch length for patch-based tokenization. |
| `Strategy` | Gets the tokenization strategy being used. |
| `Stride` | Gets the stride between patches. |
| `VocabularySize` | Gets the vocabulary size for quantization-based tokenization. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNorm(Tensor<>)` | Applies instance normalization (RevIN) to a time series. |
| `CreatePatches(Tensor<>)` | Tokenizes a time series into patches. |
| `Dequantize(Int32[],Double[])` | Dequantizes token indices back to continuous values using bin centers. |
| `Quantize(Tensor<>)` | Quantizes continuous values into discrete tokens using uniform binning. |
| `ReconstructFromPatches(List<Tensor<>>)` | Reconstructs a time series from patches (inverse of CreatePatches for non-overlapping case). |
| `ReverseInstanceNorm(Tensor<>,,)` | Reverses instance normalization on a prediction. |

