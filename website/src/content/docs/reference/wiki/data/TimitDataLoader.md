---
title: "TimitDataLoader<T>"
description: "Loads the TIMIT acoustic-phonetic continuous-speech corpus (Garofolo et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio.Benchmarks`

Loads the TIMIT acoustic-phonetic continuous-speech corpus (Garofolo et al. 1993).

## How It Works

Expects the canonical LDC distribution layout (manually extracted —
requires LDC membership; no auto-download):

Features Tensor[N, MaxTextLength] (encoded transcript); labels
Tensor[N, MaxAudioSamples] waveform. The .txt files contain a single
sentence per utterance; the WAV files are NIST sphere format originally
but most distributions ship standard PCM WAV (this loader assumes PCM).

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

