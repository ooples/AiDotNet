---
title: "GtzanDataLoader<T>"
description: "Loads the GTZAN music genre classification dataset (Tzanetakis & Cook 2002)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio.Benchmarks`

Loads the GTZAN music genre classification dataset (Tzanetakis & Cook 2002).

## How It Works

Expects the canonical Marsyas mirror layout:

Auto-download fetches the canonical Marsyas tarball.
Per-class deterministic split: first `TrainFraction` of the 100
clips per genre go to train, the rest to test/validation.
Features Tensor[N, Samples]; labels Tensor[N, 10] one-hot.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

