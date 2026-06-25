---
title: "PostprocessingRegistry<T, TOutput>"
description: "Global registry for the postprocessing pipeline."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Postprocessing`

Global registry for the postprocessing pipeline.

## For Beginners

You don't need to interact with this directly — just use AiModelBuilder:

## How It Works

**Deprecated:** This static registry causes race conditions when multiple
`AiModelBuilder` instances build models concurrently, because they overwrite
each other's pipeline. Use instance-based postprocessing via
`AiModelBuilder.ConfigurePostprocessing()` instead, which stores the pipeline
per-builder and flows it to `AiModelResult`.

## Properties

| Property | Summary |
|:-----|:--------|
| `Current` | Gets or sets the current postprocessing pipeline. |
| `IsConfigured` | Gets whether a postprocessing pipeline is currently configured. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clear` | Clears the current postprocessing pipeline. |
| `FitTransform()` | Fits the current postprocessing pipeline to data and transforms it. |
| `Transform()` | Transforms output data using the current postprocessing pipeline. |

