---
title: "CVFold<T>"
description: "Represents a single cross-validation fold with train and validation data."
section: "API Reference"
---

`Structs` · `AiDotNet.Evaluation.CrossValidation`

Represents a single cross-validation fold with train and validation data.

## Properties

| Property | Summary |
|:-----|:--------|
| `FoldIndex` | The fold number (0-indexed). |
| `TrainIndices` | Indices for training samples. |
| `TrainSize` | Number of training samples in this fold. |
| `ValidationIndices` | Indices for validation samples. |
| `ValidationSize` | Number of validation samples in this fold. |

