---
title: "GenreClassifierOptions"
description: "Options for genre classification."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Classification`

Options for genre classification.

## For Beginners

These options configure the GenreClassifier model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GenreClassifierOptions` | Initializes a new instance with default values. |
| `GenreClassifierOptions(GenreClassifierOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CustomGenres` | Custom genre labels (optional). |
| `FftSize` | FFT size. |
| `HopLength` | Hop length. |
| `ModelPath` | Path to ONNX model file (optional). |
| `NumMfccs` | Number of MFCCs to extract. |
| `OnnxOptions` | ONNX model options. |
| `SampleRate` | Audio sample rate. |
| `TopK` | Number of top predictions to return. |

