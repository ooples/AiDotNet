---
title: "LazyInitializationStrategy<T>"
description: "Lazy initialization strategy that defers weight allocation until first Forward() call."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Initialization`

Lazy initialization strategy that defers weight allocation until first Forward() call.

## For Beginners

Think of lazy initialization like making dinner reservations
versus cooking dinner. The reservation (lazy) is fast - the cooking happens later when you
actually need it. This makes creating networks much faster when you just want to inspect
them or compare their structures.

## How It Works

This strategy significantly speeds up network construction by not allocating or
initializing weight tensors until they are actually needed. This is particularly
useful for tests and when comparing network architectures without actually running them.

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

