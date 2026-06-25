---
title: "InitializationStrategies<T>"
description: "Provides factory methods and default instances for initialization strategies."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Initialization`

Provides factory methods and default instances for initialization strategies.

## How It Works

This static class provides convenient access to commonly used initialization strategies
as singletons, reducing memory allocations when the same strategy is used across
multiple layers or networks.

**Usage:**

## Properties

| Property | Summary |
|:-----|:--------|
| `Eager` | Gets the default eager initialization strategy. |
| `He` | He/Kaiming normal initialization. |
| `HeUniform` | He/Kaiming uniform initialization. |
| `Lazy` | Gets the default lazy initialization strategy. |
| `LeCun` | LeCun normal initialization. |
| `Orthogonal` | Orthogonal initialization. |
| `OrthogonalReLU` | Orthogonal initialization with sqrt(2) gain, optimized for ReLU networks. |
| `UniformSmall` | Small uniform initialization U(-0.05, 0.05). |
| `Zero` | Gets the zero initialization strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FromFile(String,WeightFileFormat)` | Creates a new initialization strategy that loads weights from a file. |

