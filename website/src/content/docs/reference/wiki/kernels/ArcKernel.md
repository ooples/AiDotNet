---
title: "ArcKernel<T>"
description: "Arc (Angular) Kernel based on the angle between vectors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Arc (Angular) Kernel based on the angle between vectors.

## For Beginners

The Arc Kernel (also called Angular Kernel) is based on the
angle θ between two vectors rather than their Euclidean distance.

In mathematical terms:
k(x, x') = 1 - (2/π) × arccos(cosine_similarity(x, x'))
= 1 - (2/π) × θ

Where θ is the angle between x and x' in radians.

Properties:

- Returns values in [0, 1]
- k(x, x) = 1 (same vector)
- k(x, x') = 0 when vectors are opposite (θ = π)
- k(x, x') = 0.5 when vectors are orthogonal (θ = π/2)

Unlike the Cosine Kernel, the Arc Kernel is positive semi-definite,
making it suitable for use as a GP kernel.

Applications:

- Directional data (wind direction, compass headings)
- Spherical data (locations on Earth, molecular orientations)
- Any data where only direction matters

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ArcKernel(Int32,Double)` | Initializes a new Arc Kernel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Order` | Gets the kernel order. |
| `OutputScale` | Gets the output scale. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the arc kernel value between two vectors. |
| `ComputeOrder0(Double,Double,Double)` | Order 0: J_0(θ) = 1 - θ/π |
| `ComputeOrder1(Double,Double,Double,Double)` | Order 1: J_1(θ) = (1/π) × (sin(θ) + (π - θ) × cos(θ)) |
| `ComputeOrder2(Double,Double,Double,Double)` | Order 2: J_2(θ) = (1/π) × (3 × sin(θ) × cos(θ) + (π - θ) × (1 + 2cos²(θ))) |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_order` | Order of the arc kernel. |
| `_outputScale` | Output scale factor. |

