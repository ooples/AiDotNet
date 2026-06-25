---
title: "EagerInitializationStrategy<T>"
description: "Eager initialization strategy that initializes weights immediately on construction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Initialization`

Eager initialization strategy that initializes weights immediately on construction.

## How It Works

This is the traditional initialization approach where weights are allocated and
initialized during layer construction. This ensures all weights are ready before
any training or inference begins.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EagerInitializationStrategy` | Initializes a new instance using the framework's default non-deterministic RNG. |
| `EagerInitializationStrategy(Random)` | Initializes a new instance using the supplied `Random` for reproducible weight initialization. |

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

