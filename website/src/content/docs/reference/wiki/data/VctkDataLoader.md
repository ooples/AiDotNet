---
title: "VctkDataLoader<T>"
description: "Loads the VCTK Corpus 0.92 multi-speaker TTS dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio.Benchmarks`

Loads the VCTK Corpus 0.92 multi-speaker TTS dataset.

## How It Works

Expects:

Auto-downloads the canonical Edinburgh DataShare zip.
Features [N, MaxTextLength] (encoded transcript); labels [N, MaxAudioSamples] waveform.
Deterministic 90/5/5 row-index split.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

