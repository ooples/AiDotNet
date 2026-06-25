---
title: "CubicRBF<T>"
description: "Implements a Cubic Radial Basis Function (RBF) that grows with the cube of the distance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RadialBasisFunctions`

Implements a Cubic Radial Basis Function (RBF) that grows with the cube of the distance.

## For Beginners

A Radial Basis Function (RBF) is a special type of mathematical function
that depends only on the distance from a center point.

The Cubic RBF is unique compared to many other RBFs because its value grows larger as you move
away from the center, rather than smaller. Specifically, it grows with the cube of the distance
(distance × distance × distance).

Think of it like a bowl shape turned upside down - the further you go from the center,
the higher the value becomes, and it grows quite rapidly with distance.

This type of function is useful in certain modeling scenarios where you expect larger values
for points that are farther away from reference points. The width parameter lets you control
how quickly the function grows with distance.

## How It Works

This class provides an implementation of a Radial Basis Function (RBF) that uses a cubic function
of the form f(r) = (r/width)³, where r is the radial distance and width is a scaling parameter.
Unlike many other RBFs that decrease with distance, the cubic RBF increases with the cube of the distance.
This makes it useful for certain regression and interpolation problems where larger responses are expected
for points farther from the centers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CubicRBF(Double)` | Initializes a new instance of the `CubicRBF` class with a specified width parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute()` | Computes the value of the Cubic Radial Basis Function for a given radius. |
| `ComputeDerivative()` | Computes the derivative of the Cubic RBF with respect to the radius. |
| `ComputeWidthDerivative()` | Computes the derivative of the Cubic RBF with respect to the width parameter. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | The numeric operations provider for type T, used for mathematical calculations. |
| `_width` | The width parameter controlling the scale of the function. |

