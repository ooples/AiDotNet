---
title: "SquaredExponentialRBF<T>"
description: "Implements a Squared Exponential (Gaussian) Radial Basis Function (RBF) of the form exp(-(er)²)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RadialBasisFunctions`

Implements a Squared Exponential (Gaussian) Radial Basis Function (RBF) of the form exp(-(er)²).

## For Beginners

A Radial Basis Function (RBF) is a special type of mathematical function
that depends only on the distance from a center point.

The Squared Exponential RBF, also commonly known as the Gaussian RBF, looks like a bell curve or
a mountain peak - it's at its highest at the center point (with a value of 1) and gradually decreases
in all directions, eventually approaching zero but never quite reaching it.

This RBF has a parameter called epsilon (e) that controls the width of the bell curve:

- A larger epsilon value creates a narrower bell curve that drops off quickly with distance
- A smaller epsilon value creates a wider bell curve that extends further

The squared exponential is the most popular RBF for many applications because:

- It's very smooth (it has derivatives of all orders)
- It has a simple mathematical form and properties
- Its shape resembles many natural processes and distributions

If you're familiar with statistics, it has the same shape as the Gaussian (normal) distribution
from probability theory.

## How It Works

This class provides an implementation of a Radial Basis Function (RBF) that uses a squared exponential form
of f(r) = exp(-(er)²), where r is the radial distance and e (epsilon) is a shape parameter
controlling the width of the function. The squared exponential RBF, also known as the Gaussian RBF,
is one of the most widely used RBFs due to its smoothness properties. It is infinitely differentiable
and has exponential decay, making it suitable for a wide range of applications in machine learning,
interpolation, and function approximation.

The squared exponential RBF corresponds to a Gaussian probability distribution and has the unique
property of being a universal approximator, meaning it can approximate any continuous function to
arbitrary precision given sufficient basis functions. It is also the only RBF that is both rotation
and translation invariant.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SquaredExponentialRBF(Double)` | Initializes a new instance of the `SquaredExponentialRBF` class with a specified shape parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute()` | Computes the value of the Squared Exponential Radial Basis Function for a given radius. |
| `ComputeDerivative()` | Computes the derivative of the Squared Exponential RBF with respect to the radius. |
| `ComputeWidthDerivative()` | Computes the derivative of the Squared Exponential RBF with respect to the shape parameter epsilon. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_epsilon` | The shape parameter (epsilon) controlling the width of the function. |
| `_numOps` | The numeric operations provider for type T, used for mathematical calculations. |

