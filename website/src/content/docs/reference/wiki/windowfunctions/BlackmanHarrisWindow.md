---
title: "BlackmanHarrisWindow<T>"
description: "Implements the Blackman-Harris window function for signal processing applications."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WindowFunctions`

Implements the Blackman-Harris window function for signal processing applications.

## For Beginners

A window function is like a special filter shape applied to your data.

The Blackman-Harris window:

- Creates a bell-shaped curve that's smoother than simpler windows
- Has very little "leakage" (unwanted frequencies) compared to other windows
- Is excellent for detecting signals that are close together in frequency

Think of it like a high-quality camera lens that gives you a clearer, more accurate picture
of the frequencies in your signal. It's especially useful in applications where you need to
distinguish between frequencies that are very close to each other, such as in audio analysis,
radar signal processing, or spectrum analysis.

## How It Works

The Blackman-Harris window is an advanced window function that provides excellent frequency 
resolution and spectral leakage suppression. It uses a weighted cosine series with four terms:
w(n) = 0.35875 - 0.48829 * cos(2pn/(N-1)) + 0.14128 * cos(4pn/(N-1)) - 0.01168 * cos(6pn/(N-1))
where n is the sample index and N is the window size.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BlackmanHarrisWindow` | Initializes a new instance of the `BlackmanHarrisWindow` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(Int32)` | Creates a Blackman-Harris window of the specified size. |
| `GetWindowFunctionType` | Gets the type identifier for this window function. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | The numeric operations provider for performing calculations with type T. |

