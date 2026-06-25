---
title: "TriangularWindow<T>"
description: "Implements the Triangular window function for signal processing applications."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WindowFunctions`

Implements the Triangular window function for signal processing applications.

## For Beginners

A window function is like a special filter that helps analyze signals more accurately.

The Triangular window:

- Creates a simple triangle shape that peaks in the middle
- Starts at 0, rises linearly to 1 at the center, then decreases linearly back to 0
- Provides a basic improvement over the Rectangular window for reducing spectral leakage

Think of it like a ramp that gradually increases and then decreases - like a mountain peak.
When analyzing frequencies in a signal, this gradual transition helps reduce some of the
unwanted artifacts that occur with the Rectangular window's abrupt edges. It's like turning
a volume knob smoothly up and down instead of suddenly flipping a switch, which creates
fewer distortions in the analysis.

## How It Works

The Triangular window is a simple window function that creates a triangular shape. It is defined by the equation:
w(n) = 1 - |2n - L|/L
where n is the sample index and L is (windowSize - 1). The Triangular window provides moderate spectral leakage
reduction compared to the Rectangular window.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TriangularWindow` | Initializes a new instance of the `TriangularWindow` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(Int32)` | Creates a Triangular window of the specified size. |
| `GetWindowFunctionType` | Gets the type identifier for this window function. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | The numeric operations provider for performing calculations with type T. |

