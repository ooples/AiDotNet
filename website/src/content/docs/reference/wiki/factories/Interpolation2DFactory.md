---
title: "Interpolation2DFactory<T>"
description: "A factory class that creates 1D interpolation functions from 2D interpolation methods."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Factories`

A factory class that creates 1D interpolation functions from 2D interpolation methods.

## For Beginners

Interpolation is a method of finding new data points within the range 
of a discrete set of known data points. It's like filling in the gaps between dots on a graph.

## How It Works

This factory helps you create interpolation functions that can estimate values between known data points.
Think of it like predicting what happens between measurements - if you know the temperature at 1pm and 3pm,
interpolation helps you estimate what it was at 2pm.

## Methods

| Method | Summary |
|:-----|:--------|
| `Create1DFromSlice(Vector<>,Vector<>,Matrix<>,Vector<>,Interpolation2DType,,Boolean,IKernelFunction<>,IMatrixDecomposition<>)` | Creates a 1D interpolation function from a 2D interpolation by fixing one coordinate. |

