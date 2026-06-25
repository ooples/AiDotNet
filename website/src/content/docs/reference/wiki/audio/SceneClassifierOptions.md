---
title: "SceneClassifierOptions"
description: "Options for acoustic scene classification."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Classification`

Options for acoustic scene classification.

## For Beginners

These options configure the SceneClassifier model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SceneClassifierOptions` | Initializes a new instance with default values. |
| `SceneClassifierOptions(SceneClassifierOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CustomScenes` | Custom scene labels (optional). |
| `FftSize` | FFT size. |
| `HopLength` | Hop length. |
| `ModelPath` | Path to ONNX model file (optional). |
| `NumMels` | Number of mel bands. |
| `NumMfccs` | Number of MFCCs. |
| `OnnxOptions` | ONNX model options. |
| `SampleRate` | Audio sample rate. |
| `TopK` | Number of top predictions to return. |

