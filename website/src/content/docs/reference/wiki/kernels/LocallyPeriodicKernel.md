---
title: "LocallyPeriodicKernel<T>"
description: "Implements the Locally Periodic kernel for measuring similarity between data points with periodic patterns."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Locally Periodic kernel for measuring similarity between data points with periodic patterns.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Locally Periodic kernel is specially designed for data that repeats in patterns (like seasonal sales,
daily temperature cycles, or sound waves) but where the pattern might change gradually over time.

## How It Works

The Locally Periodic kernel combines periodic behavior with local smoothness, making it ideal for
modeling data that shows repeating patterns that may change or decay over time or space.

Think of this kernel as a "pattern detector" that recognizes when two points are at the same phase
of a repeating cycle. For example, in temperature data, it would recognize that 3 PM on one day is
similar to 3 PM on another day (because they're at the same point in the daily cycle), but this
similarity would decrease if the days are far apart (like comparing summer to winter).

This kernel has three important parameters:

- The period controls how long each cycle is (like 24 hours for daily patterns)
- The length scale controls how quickly the similarity decays over multiple cycles
- The amplitude controls the overall strength of the pattern

The Locally Periodic kernel is particularly useful for time series forecasting, signal processing,
and any application where you need to model repeating patterns that evolve over time.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LocallyPeriodicKernel(,,)` | Initializes a new instance of the Locally Periodic kernel with optional parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Locally Periodic kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_amplitude` | Controls the overall strength or magnitude of the kernel output. |
| `_lengthScale` | Controls how quickly the similarity decays as points get farther apart in multiple cycles. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_period` | Defines the length of one complete cycle in the periodic pattern. |

