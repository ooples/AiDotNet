---
title: "FlatTopWindow<T>"
description: "Implements the Flat Top window function for signal processing applications."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WindowFunctions`

Implements the Flat Top window function for signal processing applications.

## For Beginners

A window function is a mathematical tool that helps analyze signals more accurately.

The Flat Top window:

- Has a unique shape with a flat middle section and steep sides
- Is specifically designed for accurate measurement of signal amplitudes
- Sacrifices frequency precision for amplitude precision

Imagine you're trying to measure the height of people in a crowd. Some window functions might 
give you a good idea of how many people are of each height (frequency resolution), but the 
Flat Top window specializes in telling you exactly how tall each person is (amplitude accuracy). 
This makes it ideal for calibration, testing, and situations where you need to know the exact 
strength of a signal component rather than just detecting its presence.

## How It Works

The Flat Top window is a specialized window function designed for amplitude accuracy in spectral analysis.
It uses a weighted sum of cosine terms:
w(n) = 1.0 - 1.93 * cos(2pn/(N-1)) + 1.29 * cos(4pn/(N-1)) - 0.388 * cos(6pn/(N-1)) + 0.028 * cos(8pn/(N-1))
where n is the sample index and N is the window size.
The Flat Top window has superior amplitude accuracy but poorer frequency resolution compared to other windows.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FlatTopWindow` | Initializes a new instance of the `FlatTopWindow` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(Int32)` | Creates a Flat Top window of the specified size. |
| `GetWindowFunctionType` | Gets the type identifier for this window function. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | The numeric operations provider for performing calculations with type T. |

