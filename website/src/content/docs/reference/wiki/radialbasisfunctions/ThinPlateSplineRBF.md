---
title: "ThinPlateSplineRBF<T>"
description: "Implements a Thin Plate Spline Radial Basis Function (RBF) of the form r² log(r)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RadialBasisFunctions`

Implements a Thin Plate Spline Radial Basis Function (RBF) of the form r² log(r).

## For Beginners

A Radial Basis Function (RBF) is a special type of mathematical function
that depends only on the distance from a center point.

The Thin Plate Spline RBF is named after the physical behavior of a thin metal plate that bends
under pressure. Imagine pressing down on a thin sheet of metal at specific points - the way the
sheet bends to smoothly connect those points is similar to how this function behaves.

This RBF has some unique properties:

- Unlike most RBFs that decrease with distance, the thin plate spline actually grows as you move away from the center
- It equals exactly 0 at the center point (r = 0)
- It doesn't have a width parameter like most other RBFs, making it "scale-invariant"
- It creates very smooth interpolations with minimal unnecessary wiggles or oscillations

The thin plate spline is particularly useful for 2D interpolation problems, like reconstructing
a surface from a set of scattered points, or for image warping and morphing in computer graphics.

## How It Works

This class provides an implementation of a Thin Plate Spline Radial Basis Function, which is defined as
f(r) = r² log(r), where r is the radial distance. This is a special case of the polyharmonic spline with k = 2.
The Thin Plate Spline RBF does not have a width parameter, making it scale-invariant.

The name "Thin Plate Spline" comes from the physical analogy of bending a thin sheet of metal. This RBF
minimizes a measure of energy that approximates the bending energy of a thin metal plate. It is particularly
useful for interpolation problems in two dimensions and provides a smooth interpolation that avoids
unnecessary oscillations.

Unlike many other RBFs, the thin plate spline grows with distance rather than decaying. At r = 0, the
function and its first derivative are both 0, providing a high degree of smoothness at the origin.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ThinPlateSplineRBF` | Initializes a new instance of the `ThinPlateSplineRBF` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDerivative()` | Computes the derivative of the Thin Plate Spline RBF with respect to the radius. |
| `ComputeWidthDerivative()` | Computes the derivative of the Thin Plate Spline RBF with respect to a width parameter. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | The numeric operations provider for type T, used for mathematical calculations. |

