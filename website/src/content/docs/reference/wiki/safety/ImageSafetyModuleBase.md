---
title: "ImageSafetyModuleBase<T>"
description: "Abstract base class for image safety modules."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Image`

Abstract base class for image safety modules.

## For Beginners

This base class handles the plumbing so that each image safety
module only needs to implement one method: `EvaluateImage(Tensor<T>)`.

## How It Works

Provides shared infrastructure for all image safety modules. Concrete modules implement
`Tensor{` and this base class handles the
`Vector{` bridge by reshaping vectors into tensors.

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Vector<>)` |  |
| `EvaluateImage(Tensor<>)` |  |

