---
title: "InferenceSession<T, TInput, TOutput>"
description: "Facade-friendly inference session that owns stateful inference internals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Facade-friendly inference session that owns stateful inference internals.

## How It Works

This type intentionally keeps inference internals behind the facade. Users create sequences via
`CreateSequence` and run inference via `Predict(`.

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateSequence` | Creates an independent sequence within this session. |
| `Dispose` |  |

