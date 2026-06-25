---
title: "SourceSeparationOptions"
description: "Options for music source separation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.SourceSeparation`

Options for music source separation.

## For Beginners

These options configure the SourceSeparation model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SourceSeparationOptions` | Initializes a new instance with default values. |
| `SourceSeparationOptions(SourceSeparationOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FftSize` | FFT size. |
| `HopLength` | Hop length between frames. |
| `HpssKernelSize` | HPSS kernel size for spectral separation. |
| `ModelPath` | Path to ONNX model file (optional). |
| `OnnxOptions` | ONNX model options. |
| `SampleRate` | Audio sample rate. |
| `StemCount` | Number of stems to separate (2, 4, or 5). |

