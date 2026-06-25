---
title: "Flowers102DataLoader<T>"
description: "Loads the Oxford Flowers-102 dataset (Nilsback & Zisserman 2008) — 102 fine-grained flower species."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the Oxford Flowers-102 dataset (Nilsback & Zisserman 2008) — 102 fine-grained flower species.

## How It Works

Expects:

Auto-downloads the canonical Oxford VGG release (3 separate files).

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `ReadIntVector(IMatFile,String)` | Reads an int-typed MAT variable of any width (uint8/int16/int32/double) as int[]. |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

