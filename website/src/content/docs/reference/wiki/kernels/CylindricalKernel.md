---
title: "CylindricalKernel<T>"
description: "Cylindrical Kernel for Bayesian optimization with periodic/angular dimensions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Cylindrical Kernel for Bayesian optimization with periodic/angular dimensions.

## For Beginners

The Cylindrical Kernel is designed for data that has both
"regular" dimensions (like temperature, pressure) and "angular" or "periodic"
dimensions (like angles, time of day, day of week).

For regular dimensions: Uses a standard kernel (e.g., RBF)
For angular dimensions: Uses a periodic kernel that wraps around

Example use cases:

- Optimizing chemical reactions: Temperature (regular) + catalyst angle (periodic)
- Time series: Date (regular) + hour of day (periodic)
- Robotics: Position (regular) + joint angles (periodic)

The kernel combines:
k(x, x') = k_regular(x_reg, x'_reg) × k_angular(x_ang, x'_ang)

For angular dimensions, it uses:
k_angular(θ, θ') = exp(-2 × sin²(π(θ - θ')/period) / l²)

This ensures smooth wrapping at boundaries (e.g., 359° is close to 1°).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CylindricalKernel(IKernelFunction<>,Int32[],Int32[],Double[],Double[])` | Initializes a new Cylindrical Kernel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AngularDims` | Gets the angular dimension indices. |
| `RegularDims` | Gets the regular dimension indices. |
| `RegularKernel` | Gets the regular kernel. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the cylindrical kernel value. |
| `ExtractDims(Vector<>,Int32[])` | Extracts specified dimensions from a vector. |
| `WithRBF(Int32,Int32[],Double,Double[],Double[])` | Creates a Cylindrical Kernel with RBF for regular dimensions. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_angularDims` | Indices of angular/periodic dimensions. |
| `_angularLengthScales` | Length scales for angular dimensions. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_periods` | Periods for each angular dimension. |
| `_regularDims` | Indices of regular dimensions. |
| `_regularKernel` | The kernel for regular (non-periodic) dimensions. |

