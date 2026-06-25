---
title: "IParameterizable<T, TInput, TOutput>"
description: "Interface for models that have optimizable parameters."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for models that have optimizable parameters.

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters in the model. |
| `SupportsParameterInitialization` | Gets whether this model supports direct parameter-based initialization. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetParameterChunks` | Yields the model's trainable weight tensors as references — zero-copy, streaming. |
| `GetParameters` | Gets the parameters that can be optimized. |
| `SanitizeParameters(Vector<>)` | Sanitizes random parameters to satisfy model-specific constraints. |
| `SetParameterChunks(IEnumerable<Tensor<>>)` | Streaming counterpart to `Vector{`: assigns the model's trainable weight tensors from a sequence of per-tensor chunks supplied in the SAME order `GetParameterChunks` yields them, WITHOUT ever materializing a flat `Vector<T>` of all paramete… |
| `SetParameters(Vector<>)` | Sets the model parameters. |
| `WithParameters(Vector<>)` | Creates a new instance with the specified parameters. |

