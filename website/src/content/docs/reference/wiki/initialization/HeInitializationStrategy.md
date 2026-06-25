---
title: "HeInitializationStrategy<T>"
description: "He/Kaiming initialization strategy for ReLU-family activations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Initialization`

He/Kaiming initialization strategy for ReLU-family activations.

## For Beginners

Use this when your network uses ReLU or similar activations
(which is most modern networks). It's the PyTorch default for convolutional layers.

## How It Works

He initialization accounts for the fact that ReLU zeros out half the values,
requiring larger initial weights to maintain variance through the network.
This is the recommended strategy for networks using ReLU, Leaky ReLU, GELU, or SiLU.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HeInitializationStrategy(Boolean)` | Creates a He initialization strategy. |
| `HeInitializationStrategy(Random,Boolean)` | Creates a He initialization strategy with a caller-supplied `Random` source. |

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
| `WithSeededRandom(Random)` |  |

