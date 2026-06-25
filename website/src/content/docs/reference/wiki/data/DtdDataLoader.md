---
title: "DtdDataLoader<T>"
description: "Loads the Describable Textures Dataset (DTD; Cimpoi et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the Describable Textures Dataset (DTD; Cimpoi et al. 2014) — 47 texture classes.

## How It Works

Expects:

where N is the split index (1..10). Auto-downloads the canonical
VGG tarball.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

