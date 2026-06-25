---
title: "MultiquadricRBF<T>"
description: "Implements a Multiquadric Radial Basis Function (RBF) of the form v(r² + e²)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RadialBasisFunctions`

Implements a Multiquadric Radial Basis Function (RBF) of the form v(r² + e²).

## For Beginners

A Radial Basis Function (RBF) is a special type of mathematical function
that depends only on the distance from a center point.

The Multiquadric RBF is unusual compared to most other RBFs because its value grows larger as you move
away from the center, rather than smaller. It looks like a cone with a rounded bottom - starting from
a value of e at the center point (r = 0) and gradually increasing in all directions.

This RBF has a parameter called epsilon (e) that controls the shape of the function:

- A larger epsilon value creates a flatter, more rounded shape near the center
- A smaller epsilon value creates a sharper, more pointed shape near the center

The multiquadric function is useful in certain interpolation problems where its growth properties
can lead to better numerical stability. However, this same growth property means it's often used
in combination with other techniques when working with large datasets.

## How It Works

This class provides an implementation of a Radial Basis Function (RBF) that uses a multiquadric form
of f(r) = v(r² + e²), where r is the radial distance and e (epsilon) is a shape parameter
controlling the width of the function. The multiquadric RBF is infinitely differentiable and
increases with distance, unlike many other RBFs that decrease with distance. It was introduced by
R.L. Hardy and is often used in scattered data interpolation and solving partial differential equations.

A notable property of the multiquadric RBF is that it grows with distance rather than decaying,
which can lead to better conditioning in certain interpolation problems. However, this growth also
means that the corresponding interpolation matrices can be ill-conditioned for large datasets unless
appropriate precautions are taken.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiquadricRBF(Double)` | Initializes a new instance of the `MultiquadricRBF` class with a specified shape parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute()` | Computes the value of the Multiquadric Radial Basis Function for a given radius. |
| `ComputeDerivative()` | Computes the derivative of the Multiquadric RBF with respect to the radius. |
| `ComputeWidthDerivative()` | Computes the derivative of the Multiquadric RBF with respect to the shape parameter epsilon. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_epsilon` | The shape parameter (epsilon) controlling the width of the function. |
| `_numOps` | The numeric operations provider for type T, used for mathematical calculations. |

