---
title: "GaussianWindow<T>"
description: "Implements the Gaussian window function for signal processing applications."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WindowFunctions`

Implements the Gaussian window function for signal processing applications.

## For Beginners

A window function is like a special filter that helps analyze signals more accurately.

The Gaussian window:

- Creates a smooth bell-shaped curve (like the famous "bell curve" in statistics)
- Has a shape that can be adjusted using the sigma parameter
- Provides a good balance between time and frequency resolution

Think of it like a spotlight with adjustable focus. A narrow spotlight (small sigma) gives you 
very precise location information but less overall visibility. A wide spotlight (large sigma) 
shows you more of the scene but with less precision about exact locations. The Gaussian window 
works the same way when analyzing frequencies in signals - you can adjust it to balance between 
precise frequency measurement and detecting the presence of multiple frequencies.

## How It Works

The Gaussian window is based on the Gaussian (normal) distribution and provides excellent 
time-frequency localization. It is defined by the equation:
w(n) = exp(-(n-N/2)²/(2s²))
where n is the sample index, N is the window size, and s (sigma) controls the width of the window.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaussianWindow(Double)` | Initializes a new instance of the `GaussianWindow` class with the specified sigma value. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(Int32)` | Creates a Gaussian window of the specified size. |
| `GetWindowFunctionType` | Gets the type identifier for this window function. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | The numeric operations provider for performing calculations with type T. |
| `_sigma` | The standard deviation (sigma) parameter that controls the width of the Gaussian window. |

