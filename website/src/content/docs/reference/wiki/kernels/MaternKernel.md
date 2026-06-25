---
title: "MaternKernel<T>"
description: "Implements the Matérn family of kernels with configurable smoothness parameter."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Matérn family of kernels with configurable smoothness parameter.

## For Beginners

The Matérn kernel is one of the most important kernels in GP practice.
It provides a middle ground between the very smooth RBF kernel and rougher kernels.

The smoothness parameter ν controls how "wiggly" the functions can be:

- ν = 1/2: Exponential kernel (functions are continuous but not differentiable)
- ν = 3/2: Once differentiable (good for many real-world applications)
- ν = 5/2: Twice differentiable (often used as default)
- ν → ∞: RBF/Gaussian kernel (infinitely differentiable)

In mathematical terms:
k(r) = (2^(1-ν)/Γ(ν)) × (√(2ν)×r/l)^ν × K_ν(√(2ν)×r/l)

Where:

- r = |x - x'| is the distance
- l is the length scale
- ν is the smoothness parameter
- K_ν is the modified Bessel function

## How It Works

Why use Matérn over RBF?

1. **More realistic**: Real-world functions are rarely infinitely smooth
2. **Better extrapolation**: RBF can be overly smooth for extrapolation
3. **Physical motivation**: Many physical processes have finite smoothness
4. **Computational**: ν = 1/2, 3/2, 5/2 have simple closed forms

Rule of thumb:

- If you're unsure, start with ν = 5/2 (Matérn 5/2)
- For rough data, try ν = 3/2
- For very noisy data, ν = 1/2 might help

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaternKernel(Double,Double,Double)` | Initializes a new Matérn kernel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LengthScale` | Gets the length scale. |
| `Nu` | Gets the smoothness parameter (ν). |
| `Variance` | Gets the signal variance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApproximateBesselK(Double,Double)` | Approximates the modified Bessel function K_nu(x). |
| `BesselKHalfInteger(Int32,Double)` | Computes K_{n+1/2} using closed form for half-integer orders. |
| `BesselKMiller(Double,Double)` | Miller's algorithm for modified Bessel function K_nu. |
| `Calculate(Vector<>,Vector<>)` | Calculates the Matérn kernel value between two vectors. |
| `ComputeGeneralMatern(Double)` | Computes the general Matérn function using Bessel function approximation. |
| `ComputeMatern(Double)` | Computes the Matérn function value for a given distance. |
| `GammaFunction(Double)` | Approximates the Gamma function. |
| `Matern12(Double,Double)` | Creates a Matérn 1/2 (exponential) kernel. |
| `Matern32(Double,Double)` | Creates a Matérn 3/2 kernel. |
| `Matern52(Double,Double)` | Creates a Matérn 5/2 kernel. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lengthScale` | The length scale parameter. |
| `_nu` | The smoothness parameter (ν). |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_variance` | The signal variance (kernel scale). |

