---
title: "AudioAugmenterBase<T>"
description: "Base class for audio data augmentations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Augmentation.Audio`

Base class for audio data augmentations.

## For Beginners

Audio augmentation transforms sound data to improve model
robustness to variations in recording conditions, speaking styles, and environmental noise.
Common techniques include:

- Time stretching (faster/slower without pitch change)
- Pitch shifting (higher/lower without speed change)
- Adding background noise
- Volume changes
- Time shifting (moving audio forward/backward)

## How It Works

Audio data is typically represented as a 1D waveform tensor or 2D spectrogram.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioAugmenterBase(Double,Int32)` | Initializes a new audio augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SampleRate` | Gets or sets the sample rate of the audio data in Hz. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetDuration(Tensor<>)` | Gets the duration of the audio in seconds. |
| `GetParameters` |  |
| `GetSampleCount(Tensor<>)` | Gets the number of audio samples. |

