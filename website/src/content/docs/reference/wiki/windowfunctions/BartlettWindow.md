---
title: "BartlettWindow<T>"
description: "Implements the Bartlett window function (triangular window) for signal processing applications."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WindowFunctions`

Implements the Bartlett window function (triangular window) for signal processing applications.

## For Beginners

A window function is like a special filter that smooths out a signal.

The Bartlett window specifically creates a triangle shape:

- It starts at 0 at the beginning
- Increases to 1 at the middle
- Decreases back to 0 at the end

This window is useful when you need to:

- Analyze frequencies in audio or other signals
- Reduce sharp transitions at the edges of your data
- Apply a simple, computationally efficient smoothing effect

For example, if you're analyzing sound and want to focus on a specific time segment,
the Bartlett window helps blend the edges smoothly instead of creating an abrupt cutoff.

## How It Works

The Bartlett window is a triangular window function that provides a simple approach to smoothing signals.
It is defined by the equation: w(n) = 1 - |2(n - (N-1)/2)/(N-1)|
where n is the sample index and N is the window size.
The Bartlett window has a value of 0 at both endpoints and a value of 1 at the center.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BartlettWindow` | Initializes a new instance of the `BartlettWindow` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(Int32)` | Creates a Bartlett window of the specified size. |
| `GetWindowFunctionType` | Gets the type identifier for this window function. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | The numeric operations provider for performing calculations with type T. |

