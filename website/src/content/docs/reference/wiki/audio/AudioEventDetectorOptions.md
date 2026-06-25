---
title: "AudioEventDetectorOptions"
description: "Options for audio event detection."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Classification`

Options for audio event detection.

## For Beginners

These options configure the AudioEventDetector model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioEventDetectorOptions` | Initializes a new instance with default values. |
| `AudioEventDetectorOptions(AudioEventDetectorOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CustomLabels` | Custom event labels (optional). |
| `FMax` | Maximum frequency for mel filterbank. |
| `FMin` | Minimum frequency for mel filterbank. |
| `FftSize` | FFT size. |
| `HopLength` | Hop length. |
| `ModelPath` | Path to ONNX model file (optional). |
| `NumMels` | Number of mel bands. |
| `OnnxOptions` | ONNX model options. |
| `SampleRate` | Audio sample rate. |
| `Threshold` | Confidence threshold for event detection. |
| `WindowOverlap` | Window overlap ratio (0-1). |
| `WindowSize` | Window size in seconds. |

