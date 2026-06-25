---
title: "ILoRAConfiguration<T>"
description: "Interface for configuring how LoRA (Low-Rank Adaptation) should be applied to neural network layers."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for configuring how LoRA (Low-Rank Adaptation) should be applied to neural network layers.

## For Beginners

This interface lets you define a "strategy" for how LoRA should be applied
to your model. Different strategies might:

- Apply LoRA to all dense layers
- Apply LoRA only to layers with names matching a pattern
- Apply LoRA to all layers above a certain size
- Apply different LoRA ranks to different layer types

This gives you flexible control over how your model is adapted without hardcoding the logic.

## How It Works

This interface defines a strategy pattern for applying LoRA adaptations to layers within a model.
Different implementations can provide different strategies for which layers to adapt and how.

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the scaling factor (alpha) for LoRA adaptations. |
| `FreezeBaseLayer` | Gets whether base layers should be frozen during training. |
| `Rank` | Gets the rank of the low-rank decomposition to use for adapted layers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyLoRA(ILayer<>)` | Applies LoRA adaptation to a layer if applicable according to this configuration strategy. |

