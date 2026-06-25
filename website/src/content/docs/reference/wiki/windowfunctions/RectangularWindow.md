---
title: "RectangularWindow<T>"
description: "Implements the Rectangular window function (also known as the boxcar or Dirichlet window) for signal processing applications."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WindowFunctions`

Implements the Rectangular window function (also known as the boxcar or Dirichlet window) for signal processing applications.

## For Beginners

A window function is like a special filter that helps analyze signals more accurately.

The Rectangular window:

- Is the simplest window function - it has a value of 1 for all points
- Does not taper the signal at the edges like other windows do
- Preserves the original amplitude of the signal at all points

Think of it like looking at your data through a simple rectangular frame with sharp edges.
While this window preserves the most signal energy, the abrupt transitions at the edges can
create artifacts in frequency analysis. It's like suddenly turning a speaker on and off instead
of gradually adjusting the volume - the sudden change creates additional frequency components
that weren't in the original signal.

## How It Works

The Rectangular window is the simplest window function, defined by the equation:
w(n) = 1 for all n
where n is the sample index. Unlike other window functions, the Rectangular window does not
taper or modify the signal at the edges, which can lead to spectral leakage in frequency analysis.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RectangularWindow` | Initializes a new instance of the `RectangularWindow` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(Int32)` | Creates a Rectangular window of the specified size. |
| `GetWindowFunctionType` | Gets the type identifier for this window function. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | The numeric operations provider for performing calculations with type T. |

