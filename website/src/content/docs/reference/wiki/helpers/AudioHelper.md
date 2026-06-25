---
title: "AudioHelper<T>"
description: "Helper class for loading and saving audio as tensors."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Helper class for loading and saving audio as tensors.

## For Beginners

This class converts audio files into tensors for neural networks.
Audio is loaded as [channels, samples] or [batch, channels, samples] tensors.
Values are normalized to [-1, 1] range by default.

## How It Works

Supports common audio formats without external dependencies:

- WAV: Uncompressed PCM audio (most common for ML)
- RAW: Raw PCM samples with specified parameters

## Methods

| Method | Summary |
|:-----|:--------|
| `LoadAudio(String,Boolean,Nullable<Int32>)` | Loads an audio file and returns it as a tensor with metadata. |
| `LoadRaw(String,Int32,Int32,Int32,Boolean)` | Loads raw PCM audio data. |
| `LoadWav(String,Boolean)` | Loads a WAV audio file. |
| `Resample(Tensor<>,Int32,Int32)` | Resamples audio to a different sample rate using linear interpolation. |
| `SaveWav(Tensor<>,String,Int32,Int32,Boolean)` | Saves audio tensor as a WAV file. |
| `ToMono(Tensor<>)` | Converts stereo audio to mono by averaging channels. |

