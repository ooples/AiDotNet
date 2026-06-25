---
title: "PrototypeAdamOptimizer<T>"
description: "Prototype Adam optimizer using vectorized operations via PrototypeVector."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Prototypes`

Prototype Adam optimizer using vectorized operations via PrototypeVector.
Demonstrates GPU acceleration through the Execution Engine pattern.

## How It Works

This is a PROTOTYPE for Phase A validation. The production version will integrate
with the existing AdamOptimizer class.

Key Difference from Current AdamOptimizer:

- BEFORE: Element-wise for-loops (CPU only, slow)
- AFTER: Vectorized operations (GPU accelerated when using float)

Example:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrototypeAdamOptimizer(Double,Double,Double,Double)` | Initializes a new instance of the PrototypeAdamOptimizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `TimeStep` | Gets the current time step (number of updates performed). |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateScalarVector(Int32,)` | Helper method to create a vector filled with a scalar value. |
| `Reset` | Resets the optimizer state (clears moment estimates). |
| `ToString` | Returns a string representation of the optimizer configuration. |
| `UpdateParameters(PrototypeVector<>,PrototypeVector<>)` | Updates parameters using the Adam optimization algorithm with vectorized operations. |

