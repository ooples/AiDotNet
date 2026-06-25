---
title: "BesselKernel<T>"
description: "Implements the Bessel kernel function for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Bessel kernel function for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Bessel kernel is a specialized similarity measure that works well for certain types of data,
especially data with circular or radial patterns (like sound waves, vibrations, or heat distribution
in circular objects). It's named after Friedrich Bessel, a mathematician who studied these special
mathematical functions in the 19th century.

## How It Works

The Bessel kernel is based on the Bessel functions of the first kind, which are solutions to 
Bessel's differential equation. This kernel is particularly useful for problems involving 
circular or cylindrical data patterns.

Think of the Bessel kernel as a way to compare data points while taking into account their
"wave-like" or "oscillating" properties, similar to how ripples spread in water.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BesselKernel(,)` | Initializes a new instance of the Bessel kernel with optional parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BesselFunction(,)` | Calculates the value of the Bessel function of the first kind. |
| `BesselFunctionAsymptotic(,)` | Calculates the Bessel function using an asymptotic expansion approach. |
| `BesselFunctionSeries(,)` | Calculates the Bessel function using a series expansion approach. |
| `Calculate(Vector<>,Vector<>)` | Calculates the Bessel kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_order` | The order of the Bessel function to use in the kernel calculation. |
| `_sigma` | The scaling parameter that controls the width of the kernel. |

