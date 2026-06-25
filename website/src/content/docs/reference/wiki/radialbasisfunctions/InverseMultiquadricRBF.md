---
title: "InverseMultiquadricRBF<T>"
description: "Implements an Inverse Multiquadric Radial Basis Function (RBF) of the form 1/v(r² + e²)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RadialBasisFunctions`

Implements an Inverse Multiquadric Radial Basis Function (RBF) of the form 1/v(r² + e²).

## For Beginners

A Radial Basis Function (RBF) is a special type of mathematical function
that depends only on the distance from a center point.

The Inverse Multiquadric RBF looks like an upside-down cone that flattens out at larger distances.
At the center point (r = 0), it has its highest value of 1/e, and as you move away from the center,
the function value gradually decreases toward zero, but never quite reaches it.

This RBF has a parameter called epsilon (e) that controls the shape and width of the function:

- A larger epsilon value creates a narrower peak with a faster initial drop-off
- A smaller epsilon value creates a broader peak with a more gradual initial drop-off

Unlike some other RBFs (like the Gaussian), the inverse multiquadric function has "long tails,"
meaning it decreases more slowly at larger distances. This property makes it useful for problems
where you want influence to extend further from the center points.

## How It Works

This class provides an implementation of a Radial Basis Function (RBF) that uses an inverse multiquadric form
of f(r) = 1/v(r² + e²), where r is the radial distance and e (epsilon) is a shape parameter
controlling the width of the function. The inverse multiquadric RBF is infinitely differentiable and
decreases more slowly than the Gaussian RBF as distance increases. It is often used in interpolation
problems and has good numerical properties for solving partial differential equations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InverseMultiquadricRBF(Double)` | Initializes a new instance of the `InverseMultiquadricRBF` class with a specified shape parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute()` | Computes the value of the Inverse Multiquadric Radial Basis Function for a given radius. |
| `ComputeDerivative()` | Computes the derivative of the Inverse Multiquadric RBF with respect to the radius. |
| `ComputeWidthDerivative()` | Computes the derivative of the Inverse Multiquadric RBF with respect to the shape parameter epsilon. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_epsilon` | The shape parameter (epsilon) controlling the width of the function. |
| `_numOps` | The numeric operations provider for type T, used for mathematical calculations. |

