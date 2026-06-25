---
title: "PreprocessingInfo<T, TInput, TOutput>"
description: "Stores the fitted preprocessing pipeline for inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing`

Stores the fitted preprocessing pipeline for inference.

## For Beginners

After training, your preprocessing pipeline has "learned"
things like the mean and standard deviation for scaling. This class stores all that
learned information so you can apply the same transformations to new data during
predictions.

## How It Works

This class encapsulates the preprocessing state needed to transform new data
during inference. It stores the fitted feature pipeline and optionally a target
pipeline for inverse transformation of predictions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PreprocessingInfo` | Creates a new instance of `PreprocessingInfo`. |
| `PreprocessingInfo(PreprocessingPipeline<,,>)` | Creates a new instance with the specified fitted pipeline. |
| `PreprocessingInfo(PreprocessingPipeline<,,>,PreprocessingPipeline<,,>)` | Creates a new instance with both feature and target pipelines. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsFitted` | Gets whether the feature pipeline has been fitted to data. |
| `IsTargetFitted` | Gets whether the target pipeline has been fitted to data. |
| `Pipeline` | Gets or sets the fitted feature preprocessing pipeline. |
| `TargetPipeline` | Gets or sets the fitted target preprocessing pipeline. |

## Methods

| Method | Summary |
|:-----|:--------|
| `InverseTransformPredictions()` | Inverse transforms predictions using the target pipeline. |
| `TransformFeatures()` | Transforms input features using the fitted pipeline. |

