---
title: "Caltech101DataLoader<T>"
description: "Loads the Caltech-101 image classification dataset (Fei-Fei et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the Caltech-101 image classification dataset (Fei-Fei et al. 2004).

## How It Works

Expects:

or the directory layout extracted directly under `{DataPath}`.
Auto-download fetches the canonical Caltech.edu zip.
Standard split: first `TrainImagesPerClass` images per class go
to train, the rest go to test/validation (split == Validation gets
the same set as Test since Caltech-101 has no canonical val split).

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

