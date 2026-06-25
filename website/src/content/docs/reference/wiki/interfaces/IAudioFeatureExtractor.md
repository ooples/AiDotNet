---
title: "IAudioFeatureExtractor<T>"
description: "Defines the contract for audio feature extraction algorithms."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for audio feature extraction algorithms.

## For Beginners

Audio files are just sequences of numbers representing sound waves.
Feature extractors convert these raw numbers into more useful formats:

- **MFCC**: Captures how humans perceive different frequencies
- **Chroma**: Represents musical pitch classes (C, C#, D, etc.)
- **Spectral**: Measures brightness, contrast, and other spectral properties

## How It Works

Audio feature extractors transform raw audio waveforms into meaningful representations
that can be used for tasks like speech recognition, music analysis, and audio classification.

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureDimension` | Gets the number of features produced per frame. |
| `Name` | Gets the name of this feature extractor. |
| `SampleRate` | Gets the sample rate expected by this extractor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Extract(Tensor<>)` | Extracts features from an audio waveform. |
| `Extract(Vector<>)` | Extracts features from an audio waveform. |
| `ExtractAsync(Tensor<>,CancellationToken)` | Extracts features from an audio waveform asynchronously. |

