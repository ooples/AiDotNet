---
title: "InitializationStrategyType"
description: "Specifies the type of initialization strategy to use for layer weights."
section: "API Reference"
---

`Enums` · `AiDotNet.Initialization`

Specifies the type of initialization strategy to use for layer weights.

## How It Works

Use this enum with `TrainingMemoryConfig`
to configure the default initialization strategy for all layers.

## Fields

| Field | Summary |
|:-----|:--------|
| `Eager` | Eager initialization - initialize weights immediately during layer construction. |
| `FromFile` | Load weights from an external file. |
| `Lazy` | Lazy initialization - defer weight initialization until the first forward pass. |
| `Zero` | Zero initialization - set all weights to zero. |

