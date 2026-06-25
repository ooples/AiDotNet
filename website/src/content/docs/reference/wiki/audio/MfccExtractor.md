---
title: "MfccExtractor<T>"
description: "Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio signals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Features`

Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio signals.

## For Beginners

MFCCs capture the "shape" of the audio's frequency content,
similar to how humans perceive sound. The process:

- Compute the Mel spectrogram (power spectrum on perceptual scale)
- Take the log (matches human loudness perception)
- Apply DCT (decorrelates and compresses the information)
- Keep only the first N coefficients (typically 13-40)

Why MFCCs work well for speech:

- They capture formant frequencies (vocal tract resonances)
- They're robust to background noise
- They compress audio information efficiently

Usage:

## How It Works

MFCCs are a compact representation of the spectral envelope of an audio signal.
They are widely used in speech recognition, speaker identification, and music analysis.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MfccExtractor(MfccOptions)` | Initializes a new MFCC extractor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureDimension` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Extract(Tensor<>)` |  |

