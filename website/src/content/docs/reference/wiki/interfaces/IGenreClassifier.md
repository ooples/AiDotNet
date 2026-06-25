---
title: "IGenreClassifier<T>"
description: "IGenreClassifier<T> — Interfaces in AiDotNet.Interfaces."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

_No summary documentation available yet._

## Properties

| Property | Summary |
|:-----|:--------|
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `SampleRate` | Gets the expected sample rate for input audio. |
| `SupportedGenres` | Gets the list of genres this model can classify. |
| `SupportsMultiLabel` | Gets whether this model supports multi-label classification. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Classify(Tensor<>)` | Classifies the genre of audio. |
| `ClassifyAsync(Tensor<>,CancellationToken)` | Classifies genre asynchronously. |
| `ExtractFeatures(Tensor<>)` | Extracts audio features used for classification. |
| `GetGenreProbabilities(Tensor<>)` | Gets genre probabilities for all supported genres. |
| `GetTopGenres(Tensor<>,Int32)` | Gets top-K genre predictions. |
| `TrackGenreOverTime(Tensor<>,Double)` | Tracks genre over time within a piece. |

