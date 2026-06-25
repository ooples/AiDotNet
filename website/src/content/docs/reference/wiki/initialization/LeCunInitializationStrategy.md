---
title: "LeCunInitializationStrategy<T>"
description: "LeCun initialization strategy for SELU activations and self-normalizing networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Initialization`

LeCun initialization strategy for SELU activations and self-normalizing networks.

## For Beginners

Use this with SELU activation functions for deep networks
that automatically maintain stable activations without batch normalization.

## How It Works

LeCun initialization uses variance 1/fan_in, designed for networks with SELU activation
that maintain self-normalizing properties through many layers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LeCunInitializationStrategy` | Creates a LeCun initialization strategy with the framework's default thread-safe RNG. |
| `LeCunInitializationStrategy(Random)` | Creates a LeCun initialization strategy with a caller-supplied `Random` source. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsLazy` |  |
| `LoadFromExternal` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `InitializeBiases(Tensor<>)` |  |
| `InitializeWeights(Tensor<>,Int32,Int32)` |  |

