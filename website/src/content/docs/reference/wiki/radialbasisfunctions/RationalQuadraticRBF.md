---
title: "RationalQuadraticRBF<T>"
description: "Implements a Rational Quadratic Radial Basis Function (RBF) of the form 1 - r²/(r² + e²)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RadialBasisFunctions`

Implements a Rational Quadratic Radial Basis Function (RBF) of the form 1 - r²/(r² + e²).

## For Beginners

A Radial Basis Function (RBF) is a special type of mathematical function
that depends only on the distance from a center point.

The Rational Quadratic RBF is shaped like a hill or a bell curve - it starts at its highest point
in the center (with a value of 1) and gradually decreases in all directions, eventually approaching
zero but never quite reaching it. Compared to the Gaussian RBF, it decreases more slowly as you
move away from the center, giving it "fatter tails."

This RBF has a parameter called epsilon (e) that controls the width of the hill:

- A larger epsilon value creates a wider hill that decreases more gradually with distance
- A smaller epsilon value creates a narrower hill that drops off more quickly

The rational quadratic function is useful when you want a smooth function that doesn't decay
as rapidly as the Gaussian. It can capture longer-range dependencies in your data, making it
valuable in many machine learning and spatial statistics applications.

## How It Works

This class provides an implementation of a Radial Basis Function (RBF) that uses a rational quadratic form
of f(r) = 1 - r²/(r² + e²), where r is the radial distance and e (epsilon) is a shape parameter
controlling the width of the function. The rational quadratic RBF is infinitely differentiable and
decreases from 1 at r = 0 to 0 as r approaches infinity. It has a smoother and more gradual decay
compared to the Gaussian RBF, which can be beneficial in certain applications.

This function forms a proper correlation function and has applications in various fields including
machine learning, geostatistics, and scattered data interpolation. The rational quadratic is also
related to the Student-t process in statistical modeling.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RationalQuadraticRBF(Double)` | Initializes a new instance of the `RationalQuadraticRBF` class with a specified shape parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute()` | Computes the value of the Rational Quadratic Radial Basis Function for a given radius. |
| `ComputeDerivative()` | Computes the derivative of the Rational Quadratic RBF with respect to the radius. |
| `ComputeWidthDerivative()` | Computes the derivative of the Rational Quadratic RBF with respect to the shape parameter epsilon. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_epsilon` | The shape parameter (epsilon) controlling the width of the function. |
| `_numOps` | The numeric operations provider for type T, used for mathematical calculations. |

