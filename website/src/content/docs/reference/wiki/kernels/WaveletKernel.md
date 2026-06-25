---
title: "WaveletKernel<T>"
description: "Implements the Wavelet kernel for measuring similarity between data points using wavelet functions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Wavelet kernel for measuring similarity between data points using wavelet functions.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Wavelet kernel is special because it uses wavelet functions, which are like little waves that can
detect patterns at different scales or resolutions in your data.

## How It Works

The Wavelet kernel uses wavelet functions to measure similarity between data points. Wavelets are
wave-like oscillations that start at zero, increase, and then decrease back to zero. They are
particularly useful for analyzing signals at different scales.

Think of it like this: If you're looking at a beach from far away, you might see big waves, but as you
zoom in, you'll see smaller ripples too. Wavelets can help analyze both the big waves and small ripples
in your data. The Wavelet kernel uses this property to measure similarity between data points.

The formula for the Wavelet kernel is:
k(x, y) = ? h((x_i - y_i)/a) * vc
where:

- h is the wavelet function (like the Mexican Hat wavelet)
- a is a dilation parameter that controls the width of the wavelet
- c is a scaling parameter
- ? means multiply all the results together for each dimension i

Common uses include:

- Signal processing and time series analysis
- Image processing and computer vision
- Data with multi-scale patterns
- Feature extraction at different resolutions

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WaveletKernel(IWaveletFunction<>,,)` | Initializes a new instance of the Wavelet kernel with the specified parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Wavelet kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_a` | The dilation parameter that controls the width of the wavelet. |
| `_c` | The scaling parameter that affects the overall magnitude of the kernel value. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_waveletFunction` | The wavelet function used to calculate similarity. |

