---
title: "NsynthDataLoader<T>"
description: "Loads the NSynth Neural Synth dataset (Engel et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio.Benchmarks`

Loads the NSynth Neural Synth dataset (Engel et al. 2017) — 305k musical notes
classified by instrument family.

## How It Works

Expects:

where split = train / valid / test. Auto-downloads the canonical Magenta
jsonwav.tar.gz archives. Features [N, Samples] waveforms; labels [N, 11]
one-hot instrument-family classification.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

