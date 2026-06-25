---
title: "VideoSafetyModuleBase<T>"
description: "Abstract base class for video safety modules."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Video`

Abstract base class for video safety modules.

## For Beginners

This base class handles the plumbing so that each video safety
module only needs to implement one method: `EvaluateVideo(IReadOnlyList<Tensor<T>>, double)`.

## How It Works

Provides shared infrastructure for all video safety modules. Concrete modules implement
`Double)` and this base class handles
the `Vector{` bridge.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoSafetyModuleBase(Double)` | Initializes a new video safety module base with the specified default frame rate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Vector<>)` |  |
| `EvaluateVideo(IReadOnlyList<Tensor<>>,Double)` |  |

