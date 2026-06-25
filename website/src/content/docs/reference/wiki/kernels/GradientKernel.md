---
title: "GradientKernel<T>"
description: "Kernel that incorporates gradient observations for GPs with derivative information."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Kernel that incorporates gradient observations for GPs with derivative information.

## For Beginners

Sometimes you have access not just to function values f(x),
but also to gradient information ∇f(x) (the rate of change in each direction).

This is common in:

- Bayesian optimization: When evaluating an expensive function, you might also

get gradient information "for free" (e.g., via automatic differentiation)

- Physics-informed ML: Physical laws often constrain derivatives
- Adjoint methods: Gradients are computed alongside function values

The GradientKernel extends a base kernel to model both values and gradients.
If f(x) is a GP with kernel k(x, x'), then:

- Cov(f(x), f(x')) = k(x, x')
- Cov(∂f/∂xᵢ, f(x')) = ∂k(x, x')/∂xᵢ
- Cov(∂f/∂xᵢ, ∂f/∂x'ⱼ) = ∂²k(x, x')/∂xᵢ∂x'ⱼ

This lets you use gradient observations to improve predictions, especially
when gradient observations are cheaper than function observations.

## How It Works

Usage: Create with a base kernel (RBF, Matern, etc.) that supports gradients.
The resulting kernel operates on extended vectors: [x; gradient_dim_flag; is_gradient_obs].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientKernel(Int32,GradientKernel<>.GradientKernelType,Double,Double)` | Initializes a new GradientKernel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseKernel` | Gets the base kernel. |
| `InputDim` | Gets the input dimensionality. |
| `KernelType` | Gets the kernel type. |
| `LengthScale` | Gets the length scale. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the kernel value between two points (value-value, value-gradient, or gradient-gradient). |
| `ComputeDistance(Vector<>,Vector<>)` | Computes Euclidean distance between two vectors. |
| `ComputeFirstDerivative(Vector<>,Vector<>,Int32,Boolean)` | Computes the first derivative of the kernel with respect to one input. |
| `ComputeMatern32FirstDerivative(Vector<>,Vector<>,Int32,Boolean)` | Matern 3/2 first derivative. |
| `ComputeMatern32SecondDerivative(Vector<>,Vector<>,Int32,Int32)` | Matern 3/2 second derivative. |
| `ComputeMatern52FirstDerivative(Vector<>,Vector<>,Int32,Boolean)` | Matern 5/2 first derivative. |
| `ComputeMatern52SecondDerivative(Vector<>,Vector<>,Int32,Int32)` | Matern 5/2 second derivative. |
| `ComputeRBFFirstDerivative(Double,Double,Double,Boolean)` | RBF first derivative: ∂k/∂x_i = -(x_i - x'_i)/l² × k(x, x') |
| `ComputeRBFSecondDerivative(Double,Double,Double,Int32,Int32,Double,Double)` | RBF second derivative: ∂²k/∂x_i∂x'_j = (δ_ij/l² - diff_i × diff_j/l⁴) × k(x, x') |
| `ComputeSecondDerivative(Vector<>,Vector<>,Int32,Int32)` | Computes the second derivative of the kernel (mixed partial). |
| `CreateGradientObservation(Vector<>,Int32)` | Creates an extended vector for a gradient observation. |
| `CreateValueObservation(Vector<>)` | Creates an extended vector for a value observation. |
| `ExtractInput(Vector<>)` | Extracts the input part from an extended vector (removes the gradient dimension flag). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_baseKernel` | The base kernel from which derivatives are computed. |
| `_inputDim` | The input dimensionality. |
| `_kernelType` | The kernel type for gradient computation. |
| `_lengthScale` | The length scale parameter (needed for gradient computation). |
| `_numOps` | Operations for performing numeric calculations with type T. |

