---
title: "ExponentialRBF<T>"
description: "Implements an Exponential Radial Basis Function (RBF) of the form exp(-e*r)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RadialBasisFunctions`

Implements an Exponential Radial Basis Function (RBF) of the form exp(-e*r).

## For Beginners

A Radial Basis Function (RBF) is a special type of mathematical function
that depends only on the distance from a center point.

The Exponential RBF decreases as you move away from the center point, following an exponential decay pattern.
Think of it like a hill or mountain that starts at its highest point in the center and then gradually
slopes downward in all directions, never quite reaching zero.

This specific RBF has a parameter called epsilon (e) that controls how quickly the "hill" drops off:

- A larger epsilon value creates a steeper hill that drops off quickly with distance
- A smaller epsilon value creates a more gradual slope that extends further

The exponential RBF is useful in many machine learning applications, especially when you want a smoother
decrease with distance compared to a Gaussian RBF.

## How It Works

This class provides an implementation of a Radial Basis Function (RBF) that uses an exponential decay
of the form f(r) = exp(-e*r), where r is the radial distance and e (epsilon) is a width parameter
controlling how quickly the function decreases with distance. The exponential RBF is sometimes called
the Laplacian RBF and is related to the distribution of the same name. It decreases less rapidly than
the Gaussian RBF for small distances but has a more gradual asymptotic behavior for large distances.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExponentialRBF(Double)` | Initializes a new instance of the `ExponentialRBF` class with a specified width parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute()` | Computes the value of the Exponential Radial Basis Function for a given radius. |
| `ComputeDerivative()` | Computes the derivative of the Exponential RBF with respect to the radius. |
| `ComputeWidthDerivative()` | Computes the derivative of the Exponential RBF with respect to the width parameter epsilon. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_epsilon` | The width parameter (epsilon) controlling how quickly the function decreases with distance. |
| `_numOps` | The numeric operations provider for type T, used for mathematical calculations. |

