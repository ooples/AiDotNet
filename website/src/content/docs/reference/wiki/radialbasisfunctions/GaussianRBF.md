---
title: "GaussianRBF<T>"
description: "Implements a Gaussian Radial Basis Function (RBF) of the form exp(-e*r²)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RadialBasisFunctions`

Implements a Gaussian Radial Basis Function (RBF) of the form exp(-e*r²).

## For Beginners

A Radial Basis Function (RBF) is a special type of mathematical function
that depends only on the distance from a center point.

The Gaussian RBF is shaped like a bell curve or a mountain peak - it's at its highest at the center point
and gradually decreases in all directions, eventually approaching zero. This particular RBF is named
"Gaussian" because it uses the same mathematical form as the Gaussian (normal) distribution from statistics.

This RBF has a parameter called epsilon (e) that controls the width of the bell curve:

- A larger epsilon value creates a narrower bell curve that drops off quickly with distance
- A smaller epsilon value creates a wider bell curve that extends further

The Gaussian RBF is very popular in machine learning applications like neural networks and regression
because it has nice mathematical properties and creates smooth interpolations between data points.

## How It Works

This class provides an implementation of a Radial Basis Function (RBF) that uses a Gaussian form
of f(r) = exp(-e*r²), where r is the radial distance and e (epsilon) is a width parameter
controlling how quickly the function decreases with distance. The Gaussian RBF is one of the most
widely used RBFs due to its smooth behavior and mathematical properties. It is infinitely differentiable
and has exponential decay, making it suitable for a wide range of applications in machine learning,
interpolation, and function approximation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaussianRBF(Double)` | Initializes a new instance of the `GaussianRBF` class with a specified width parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute()` | Computes the value of the Gaussian Radial Basis Function for a given radius. |
| `ComputeDerivative()` | Computes the derivative of the Gaussian RBF with respect to the radius. |
| `ComputeWidthDerivative()` | Computes the derivative of the Gaussian RBF with respect to the width parameter epsilon. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_epsilon` | The width parameter (epsilon) controlling how quickly the function decreases with distance. |
| `_numOps` | The numeric operations provider for type T, used for mathematical calculations. |

