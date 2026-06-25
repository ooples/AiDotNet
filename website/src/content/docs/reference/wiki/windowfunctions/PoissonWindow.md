---
title: "PoissonWindow<T>"
description: "Implements the Poisson window function for signal processing applications."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WindowFunctions`

Implements the Poisson window function for signal processing applications.

## For Beginners

A window function is like a special filter that helps analyze signals more accurately.

The Poisson window:

- Creates a curve that decays exponentially from the center to the edges
- Has a shape that can be adjusted using the alpha parameter
- Provides a smooth transition that helps reduce artifacts in frequency analysis

Think of the Poisson window like a spotlight that gradually fades out from the center.
The alpha parameter controls how quickly this fading happens - a higher alpha means
a faster fade out from the center. This gradual fading helps when analyzing signals by
reducing the artificial effects that occur when you're only looking at a segment of
a longer signal.

## How It Works

The Poisson window is an exponential window function defined by the equation:
w(n) = exp(-a|n-N/2|/(N/2))
where n is the sample index, N is the window size, and a (alpha) is a parameter
that controls the rate of decay from the center of the window to the edges.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PoissonWindow(Double)` | Initializes a new instance of the `PoissonWindow` class with the specified alpha value. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(Int32)` | Creates a Poisson window of the specified size. |
| `GetWindowFunctionType` | Gets the type identifier for this window function. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | The alpha parameter that controls the shape of the Poisson window. |
| `_numOps` | The numeric operations provider for performing calculations with type T. |

