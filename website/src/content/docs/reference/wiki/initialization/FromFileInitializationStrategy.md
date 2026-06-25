---
title: "FromFileInitializationStrategy<T>"
description: "Initialization strategy that loads weights from an external file."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Initialization`

Initialization strategy that loads weights from an external file.

## For Beginners

Transfer learning is like giving your network a head start
by using weights that were already trained on a similar task. Instead of starting
from random values, you start with values that already know useful patterns.

## How It Works

This strategy loads pre-trained weights from a file, enabling transfer learning
and model checkpointing. Weights are loaded during the first initialization call
and cached for subsequent layers.

Supported formats:

- JSON: Human-readable format with weight arrays
- Binary: Compact binary format for faster loading

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FromFileInitializationStrategy(String,WeightFileFormat)` | Initializes a new instance of the `FromFileInitializationStrategy` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsLazy` |  |
| `LoadFromExternal` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearCache` | Clears the cached weights, forcing a reload on next initialization. |
| `InitializeBiases(Tensor<>)` |  |
| `InitializeWeights(Tensor<>,Int32,Int32)` |  |
| `Reset` | Resets the layer indices for a fresh initialization pass. |

