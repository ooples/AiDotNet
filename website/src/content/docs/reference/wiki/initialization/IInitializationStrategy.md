---
title: "IInitializationStrategy<T>"
description: "Defines a strategy for initializing neural network layer parameters."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Initialization`

Defines a strategy for initializing neural network layer parameters.

## For Beginners

This controls how the network sets up its initial weights.

Different strategies have different trade-offs:

- Lazy initialization makes network construction fast (good for tests)
- Eager initialization is the traditional approach (slightly slower construction)
- FromFile loads pre-trained weights (for transfer learning)

## How It Works

This interface allows control over when and how layer weights are initialized.
Different strategies can be used for different use cases:

- Lazy: Defer initialization until first Forward() call (fast construction)
- Eager: Initialize immediately on construction (current behavior)
- FromFile: Load weights from a file instead of random initialization
- Zero: Initialize all weights to zero (useful for testing)

## Properties

| Property | Summary |
|:-----|:--------|
| `IsLazy` | Gets a value indicating whether this strategy defers initialization until first use. |
| `LoadFromExternal` | Gets a value indicating whether weights should be loaded from an external source. |

## Methods

| Method | Summary |
|:-----|:--------|
| `InitializeBiases(Tensor<>)` | Initializes the biases tensor with appropriate values. |
| `InitializeWeights(Tensor<>,Int32,Int32)` | Initializes the weights tensor with appropriate values. |

