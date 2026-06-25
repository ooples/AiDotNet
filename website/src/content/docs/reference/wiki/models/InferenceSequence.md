---
title: "InferenceSequence<T, TInput, TOutput>"
description: "Represents one independent, stateful inference sequence (e.g., one chat/generation stream)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents one independent, stateful inference sequence (e.g., one chat/generation stream).

## How It Works

A sequence may keep internal state across calls when inference optimizations are enabled (e.g., KV-cache).
Call `Reset` to start a new logical sequence on the same object.

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` |  |
| `Predict()` | Runs a prediction for the given input within this sequence. |
| `Reset` | Resets sequence-local inference state. |

