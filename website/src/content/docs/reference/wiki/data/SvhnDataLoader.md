---
title: "SvhnDataLoader<T>"
description: "Loads the SVHN (Street View House Numbers) Format-2 dataset (Netzer et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the SVHN (Street View House Numbers) Format-2 dataset (Netzer et al. 2011) — 32×32 RGB digits.

## How It Works

Expects:

Each .mat file contains `X` (uint8 array shaped [32, 32, 3, N], column-
major) and `y` (int8 array shaped [N, 1]) with labels 1..10 (10 = digit 0).
Auto-downloads from Stanford's canonical mirror.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `LoadSvhnMat(String)` | Reads an SVHN-format .mat file (variables `X`: uint8 4-D and `y`: int8 2-D). |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

