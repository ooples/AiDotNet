---
title: "AudioSafetyModuleBase<T>"
description: "Abstract base class for audio safety modules."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Audio`

Abstract base class for audio safety modules.

## For Beginners

This base class handles the plumbing so that each audio safety
module only needs to implement one method: `EvaluateAudio(Vector<T>, int)`.

## How It Works

Provides shared infrastructure for all audio safety modules. Concrete modules implement
`Int32)` and this base class handles the
`Vector{` bridge using the configured sample rate.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioSafetyModuleBase(Int32)` | Initializes a new audio safety module base with the specified default sample rate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Vector<>)` |  |
| `EvaluateAudio(Vector<>,Int32)` |  |

