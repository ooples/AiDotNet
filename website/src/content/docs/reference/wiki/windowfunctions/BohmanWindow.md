---
title: "BohmanWindow<T>"
description: "Implements the Bohman window function for signal processing applications."
section: "API Reference"
---

`Models & Types` · `AiDotNet.WindowFunctions`

Implements the Bohman window function for signal processing applications.

## For Beginners

A window function is like a special filter that helps analyze signals more accurately.

The Bohman window:

- Creates a smooth bell-shaped curve that touches zero at both ends
- Has excellent performance in reducing "spectral leakage" (unwanted frequency artifacts)
- Uses a more complex mathematical formula than simpler windows

Think of signal processing like trying to hear one conversation in a crowded room. 
The Bohman window is like a sophisticated noise-canceling headphone that helps you 
focus on just the conversation you want to hear, filtering out background noise very 
effectively. It's particularly useful in applications where you need to distinguish 
between frequencies that are very close to each other, such as radar systems, sonar, 
or detailed audio analysis.

## How It Works

The Bohman window is a specialized window function that provides excellent spectral characteristics
with very low sidelobe levels. It's defined by a more complex formula compared to simpler windows:
w(n) = (1 - |x|) * cos(π|x|) + (1/π) * sin(π|x|)
where x = 2n/(N-1) - 1 and n is the sample index and N is the window size.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BohmanWindow` | Initializes a new instance of the `BohmanWindow` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(Int32)` | Creates a Bohman window of the specified size. |
| `GetWindowFunctionType` | Gets the type identifier for this window function. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | The numeric operations provider for performing calculations with type T. |

