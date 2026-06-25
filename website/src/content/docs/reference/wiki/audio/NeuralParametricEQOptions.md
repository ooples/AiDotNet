---
title: "NeuralParametricEQOptions"
description: "Configuration options for the Neural Parametric EQ model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Effects`

Configuration options for the Neural Parametric EQ model.

## For Beginners

A parametric EQ lets you boost or cut specific frequency ranges
in audio (more bass, less treble, etc.). Normally you adjust this by ear, but Neural
Parametric EQ uses AI to do it automatically. Give it audio and a target sound, and
it figures out the right EQ settings.

## How It Works

Neural Parametric EQ (Steinmetz et al., 2022) uses a neural network to automatically
estimate parametric EQ settings that match a target frequency response. Instead of
manually adjusting filter bands, the model predicts optimal gain, frequency, and Q
values for each band. This is useful for automatic audio mastering, hearing aid fitting,
and frequency response matching between recordings.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderDim` | Gets or sets the encoder hidden dimension. |
| `FFTSize` | Gets or sets the FFT size for frequency analysis. |
| `GainRange` | Gets or sets the gain range in dB (-GainRange to +GainRange). |
| `HopLength` | Gets or sets the hop length for STFT. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxFrequency` | Gets or sets the maximum frequency in Hz for EQ bands. |
| `MinFrequency` | Gets or sets the minimum frequency in Hz for EQ bands. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumBands` | Gets or sets the number of EQ bands to predict. |
| `NumEncoderLayers` | Gets or sets the number of encoder layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Variant` | Gets or sets the model variant ("small", "base"). |

