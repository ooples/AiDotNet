---
title: "Stl10DataLoader<T>"
description: "Loads the STL-10 image classification dataset (Coates et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the STL-10 image classification dataset (Coates et al. 2011) at native 96×96 resolution.

## How It Works

Expects the canonical Stanford binary release:

Auto-downloads the canonical tarball. Pixel data is stored channel-major
(R·R·…G·G·…B·B·…) and column-major within a channel — this loader
transposes to standard row-major HWC during decoding.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

