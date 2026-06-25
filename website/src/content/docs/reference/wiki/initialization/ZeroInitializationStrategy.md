---
title: "ZeroInitializationStrategy<T>"
description: "Zero initialization strategy that sets all weights to zero."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Initialization`

Zero initialization strategy that sets all weights to zero.

## How It Works

This strategy initializes all weights to zero. It is primarily useful for testing
to ensure deterministic behavior, or for specific network architectures where
zero initialization is desired for certain layers.

**Warning:** Zero initialization typically should not be used for training
as it prevents symmetry breaking and leads to poor learning. Use for testing only.

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

