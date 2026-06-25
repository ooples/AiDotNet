---
title: "TextSafetyModuleBase<T>"
description: "Abstract base class for text safety modules providing common text processing utilities."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Text`

Abstract base class for text safety modules providing common text processing utilities.

## For Beginners

This base class handles the boring plumbing so that each
text safety module only needs to implement one method: `EvaluateText(string)`.

## How It Works

Provides shared infrastructure for all text safety modules, including the
vector-to-text conversion needed to satisfy `ISafetyModule`.
Concrete modules implement `String)` and this base
class handles the `Vector{` bridge.

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Vector<>)` |  |
| `EvaluateText(String)` |  |

