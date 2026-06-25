---
title: "PreprocessingRegistry<T, TInput>"
description: "Global registry for the preprocessing pipeline."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Preprocessing`

Global registry for the preprocessing pipeline.

## For Beginners

This is like a global settings panel for data preprocessing.
You don't need to interact with this directly - just use AiModelBuilder:

The configured preprocessing is automatically applied to all models.

## How It Works

**Deprecated:** This static registry causes race conditions when multiple
`AiModelBuilder` instances build models concurrently, because they overwrite
each other's pipeline. Use instance-based preprocessing via
`AiModelBuilder.ConfigurePreprocessing()` instead, which stores the pipeline
per-builder and flows it to `AiModelResult` via `PreprocessingInfo`.

PreprocessingRegistry provides a singleton pattern for managing the active preprocessing pipeline.
By default, a standard pipeline with imputation and scaling is used. Users can configure
custom preprocessing via AiModelBuilder.ConfigurePreprocessing().

## Properties

| Property | Summary |
|:-----|:--------|
| `Current` | Gets or sets the current preprocessing pipeline. |
| `IsConfigured` | Gets whether a preprocessing pipeline is currently configured. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clear` | Clears the current preprocessing pipeline. |
| `FitTransform()` | Fits the current preprocessing pipeline to data and transforms it. |
| `Transform()` | Transforms input data using the current preprocessing pipeline. |

