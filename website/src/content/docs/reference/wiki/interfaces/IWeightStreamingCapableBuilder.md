---
title: "IWeightStreamingCapableBuilder<T, TInput, TOutput>"
description: "Optional companion interface for builders that support PaLM-E-scale weight streaming."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Optional companion interface for builders that support PaLM-E-scale weight
streaming. Kept separate from `IAiModelBuilder`
so the introduction of `WeightStreamingConfig)` does NOT
break external implementers of `IAiModelBuilder`. Cast a builder to
this interface (or use the concrete `AiModelBuilder`)
to opt into the streaming control surface.

## Methods

| Method | Summary |
|:-----|:--------|
| `ConfigureWeightStreaming(WeightStreamingConfig)` | Configures weight streaming behaviour for the model under construction. |

