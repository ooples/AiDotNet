---
title: "BesselRBF<T>"
description: "Implements the Bessel Radial Basis Function based on Bessel functions of the first kind."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RadialBasisFunctions`

Implements the Bessel Radial Basis Function based on Bessel functions of the first kind.

## For Beginners

A Radial Basis Function (RBF) is a special type of mathematical function
that depends only on the distance from a center point.

The Bessel RBF uses Bessel functions, which are important in physics and engineering for problems
with circular or cylindrical symmetry. Think of them as functions that can model wave-like behavior,
like the vibrations on a circular drum or the pattern of ripples on a pond.

This particular RBF has two main parameters:

- epsilon (e): Controls the width of the function (how quickly it changes with distance)
- nu (?): Controls the order of the Bessel function (affects the shape and oscillatory behavior)

Bessel RBFs are useful when your data or problem has circular patterns or oscillatory features.

## How It Works

This class provides an implementation of a Radial Basis Function (RBF) using Bessel functions of the first kind.
The Bessel RBF is defined as J_?(e*r)/(e*r)^?, where J_? is the Bessel function of the first kind of order ?,
e is the width parameter, and r is the radial distance. This RBF is particularly useful for problems with
circular or spherical symmetry, and in cases where oscillatory behavior is expected.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BesselRBF(Double,Double)` | Initializes a new instance of the `BesselRBF` class with specified parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute()` | Computes the value of the Bessel Radial Basis Function for a given radius. |
| `ComputeDerivative()` | Computes the derivative of the Bessel RBF with respect to the radius. |
| `ComputeWidthDerivative()` | Computes the derivative of the Bessel RBF with respect to the width parameter epsilon. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_epsilon` | The width parameter (epsilon) controlling how quickly the function decreases with distance. |
| `_nu` | The order parameter (nu) of the Bessel function. |
| `_numOps` | The numeric operations provider for type T, used for mathematical calculations. |

