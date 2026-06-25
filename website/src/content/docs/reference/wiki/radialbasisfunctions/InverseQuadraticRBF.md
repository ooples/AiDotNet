---
title: "InverseQuadraticRBF<T>"
description: "Implements an Inverse Quadratic Radial Basis Function (RBF) of the form 1/(1 + (er)²)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RadialBasisFunctions`

Implements an Inverse Quadratic Radial Basis Function (RBF) of the form 1/(1 + (er)²).

## For Beginners

A Radial Basis Function (RBF) is a special type of mathematical function
that depends only on the distance from a center point.

The Inverse Quadratic RBF looks like a smooth bell-shaped curve that flattens out at larger distances.
At the center point (r = 0), it has its maximum value of 1, and as you move away from the center,
the function value gradually decreases toward zero, but never quite reaches it.

This RBF has a parameter called epsilon (e) that controls the shape and width of the function:

- A larger epsilon value creates a narrower bell curve that drops off quickly with distance
- A smaller epsilon value creates a wider bell curve that extends further

The inverse quadratic function is similar to the Gaussian RBF in shape but has "longer tails,"
meaning it decreases more slowly at larger distances. This property can be useful when you want
data points to have influence over a broader range.

## How It Works

This class provides an implementation of a Radial Basis Function (RBF) that uses an inverse quadratic form
of f(r) = 1/(1 + (er)²), where r is the radial distance and e (epsilon) is a shape parameter
controlling the width of the function. The inverse quadratic RBF is infinitely differentiable and
decreases more slowly than the Gaussian RBF but faster than the inverse multiquadric RBF as distance increases.
It has properties that make it useful for scattered data interpolation and solving differential equations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InverseQuadraticRBF(Double)` | Initializes a new instance of the `InverseQuadraticRBF` class with a specified shape parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute()` | Computes the value of the Inverse Quadratic Radial Basis Function for a given radius. |
| `ComputeDerivative()` | Computes the derivative of the Inverse Quadratic RBF with respect to the radius. |
| `ComputeWidthDerivative()` | Computes the derivative of the Inverse Quadratic RBF with respect to the shape parameter epsilon. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_epsilon` | The shape parameter (epsilon) controlling the width of the function. |
| `_numOps` | The numeric operations provider for type T, used for mathematical calculations. |

