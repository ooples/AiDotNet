---
title: "LjSpeechDataLoader<T>"
description: "Loads the LJSpeech 1.1 single-speaker TTS corpus (Ito & Johnson 2017)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio.Benchmarks`

Loads the LJSpeech 1.1 single-speaker TTS corpus (Ito & Johnson 2017).

## How It Works

Expects:

Auto-downloads the canonical keithito.com tar.bz2.
Features Tensor[N, MaxTextLength] (encoded text tokens); labels
Tensor[N, MaxAudioSamples] (audio waveform, mono, 22.05 kHz, [-1, 1]).
LJSpeech has no canonical train/val/test splits — this loader applies a
deterministic 90/5/5 by row index in the metadata file.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

