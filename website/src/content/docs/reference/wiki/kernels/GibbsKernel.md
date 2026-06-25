---
title: "GibbsKernel<T>"
description: "Implements the Gibbs kernel with input-dependent length scales for non-stationary covariance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Gibbs kernel with input-dependent length scales for non-stationary covariance.

## For Beginners

The Gibbs kernel is a non-stationary kernel where the length scale
can vary depending on where you are in input space. This allows modeling functions that
are smooth in some regions and vary rapidly in others.

In mathematical terms:
k(x, x') = √(2×l(x)×l(x') / (l(x)² + l(x')²)) × exp(-r² / (l(x)² + l(x')²))

Where:

- l(x) is the length scale function evaluated at x
- r = |x - x'| is the Euclidean distance

Unlike stationary kernels (like RBF) where the same length scale applies everywhere,
the Gibbs kernel allows:

- Tight length scales where you want to capture fine details
- Long length scales where the function is smooth

## How It Works

Applications:

- Functions that transition between smooth and rough regions
- Spatially varying correlation structures
- Time series where dynamics change over time
- Heteroscedastic (non-constant variance) modeling

Example: Stock prices might be smooth during normal trading but vary rapidly during
market opens, news events, or high volatility periods.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GibbsKernel(Func<Vector<>,Double>,Double)` | Initializes a new Gibbs kernel with a custom length scale function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Variance` | Gets the signal variance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Gibbs kernel value between two vectors. |
| `GetLengthScale(Vector<>)` | Gets the length scale at a given point. |
| `WithInterpolatedLengthScale(Double[],Double,Double,Double)` | Creates a Gibbs kernel with length scale learned from data. |
| `WithLinearLengthScale(Double,Double,Double)` | Creates a Gibbs kernel with a linear length scale function. |
| `WithSinusoidalLengthScale(Double,Double,Double,Double)` | Creates a Gibbs kernel with a sinusoidal length scale function. |
| `WithStepLengthScale(Double,Double,Double,Double,Double)` | Creates a Gibbs kernel with a step function length scale. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lengthScaleFunction` | The function that maps input locations to length scales. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_variance` | The signal variance (overall scale of the kernel). |

