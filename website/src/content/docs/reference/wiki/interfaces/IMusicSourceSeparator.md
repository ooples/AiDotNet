---
title: "IMusicSourceSeparator<T>"
description: "Interface for music source separation models that isolate individual instruments/vocals from a mix."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for music source separation models that isolate individual instruments/vocals from a mix.

## For Beginners

Source separation is like un-mixing a smoothie back into
its original fruits.

How it works:

1. The mixed audio is converted to a spectrogram
2. A neural network learns which parts belong to which source
3. Masks are applied to isolate each source
4. Individual spectrograms are converted back to audio

Common separations:

- 2-stem: Vocals vs Accompaniment
- 4-stem: Vocals, Drums, Bass, Other
- 5-stem: Vocals, Drums, Bass, Piano, Other

Use cases:

- Karaoke (remove vocals)
- Remixing (isolate and rearrange parts)
- Music transcription (analyze individual instruments)
- Sample extraction (get drum loops, vocal hooks)
- Music education (practice with isolated parts)

Popular models:

- Demucs (Facebook/Meta)
- Spleeter (Deezer)
- Open-Unmix

## How It Works

Music source separation (also called audio source separation or "unmixing") takes a
mixed audio signal and separates it into individual components like vocals, drums,
bass, and other instruments.

This interface extends `IFullModel` for Tensor-based audio processing.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `NumStems` | Gets the number of stems/sources this model produces. |
| `SampleRate` | Gets the expected sample rate for input audio. |
| `SupportedSources` | Gets the sources this model can separate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractSource(Tensor<>,String)` | Extracts a specific source from the mix. |
| `GetSourceMask(Tensor<>,String)` | Gets the soft mask for a specific source. |
| `Remix(SourceSeparationResult<>,IReadOnlyDictionary<String,Double>)` | Remixes the separated sources with custom volumes. |
| `RemoveSource(Tensor<>,String)` | Removes a specific source from the mix. |
| `Separate(Tensor<>)` | Separates all sources from the audio mix. |
| `SeparateAsync(Tensor<>,CancellationToken)` | Separates all sources asynchronously. |

