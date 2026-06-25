---
title: "ExpSineSquaredKernel<T>"
description: "Implements the Exp-Sine-Squared (Periodic) kernel for modeling repeating patterns."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Exp-Sine-Squared (Periodic) kernel for modeling repeating patterns.

## For Beginners

The Exp-Sine-Squared kernel is designed for data that has repeating patterns,
like daily temperature cycles, seasonal sales, or heartbeat signals.

In mathematical terms: k(x, x') = exp(-2 * sin²(π * |x - x'| / period) / lengthScale²)

Where:

- period: How often the pattern repeats (e.g., 24 hours for daily cycles)
- lengthScale: How smooth the pattern is within each period

## How It Works

Think of it like a wave function that repeats forever. If you have:

- Daily temperature data, period = 24 (hours)
- Yearly sales data, period = 12 (months) or 365 (days)
- Weekly patterns, period = 7 (days)

The kernel considers points at the same phase of the cycle as similar, regardless
of how many cycles apart they are. So Monday's data is similar to other Mondays,
even if they're weeks apart.

This kernel is commonly used in:

- Time series forecasting with seasonal patterns
- Signal processing with periodic components
- Any domain where you expect repeating behavior

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExpSineSquaredKernel(Double,Double)` | Initializes a new instance of the Exp-Sine-Squared (Periodic) kernel. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Exp-Sine-Squared kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lengthScale` | The length scale parameter that controls smoothness within each period. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_period` | The period of the repeating pattern. |

