---
title: "WaveRBF<T>"
description: "Implements a Wave (Sinc) Radial Basis Function (RBF) of the form sin(er)/(er)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RadialBasisFunctions`

Implements a Wave (Sinc) Radial Basis Function (RBF) of the form sin(er)/(er).

## For Beginners

A Radial Basis Function (RBF) is a special type of mathematical function
that depends only on the distance from a center point.

The Wave RBF is unique among RBFs because instead of simply decreasing with distance, it creates
a wave-like pattern that alternates between positive and negative values as you move away from
the center. Think of it like the ripples that spread out when you drop a stone in water - the
height of the water rises and falls in circles moving outward from where the stone hit.

This RBF has a parameter called epsilon (e) that controls how tightly packed these "ripples" are:

- A larger epsilon value creates more tightly packed ripples (higher frequency oscillations)
- A smaller epsilon value creates more widely spaced ripples (lower frequency oscillations)

The Wave RBF is particularly useful for modeling phenomena that naturally have wave-like properties,
such as sound, electromagnetic fields, or certain types of physical simulations.

## How It Works

This class provides an implementation of a Radial Basis Function (RBF) that uses a wave form
of f(r) = sin(er)/(er), where r is the radial distance and e (epsilon) is a shape parameter
controlling the frequency of oscillations. This function is also known as the spherical Bessel function
of the first kind of order zero, or more commonly as the "sinc" function when scaled.

Unlike most other RBFs that monotonically decrease with distance, the Wave RBF oscillates, creating
positive and negative lobes. This oscillatory behavior can be useful for modeling wave-like phenomena
or for approximating functions with periodic components. The function equals 1 at r = 0 and approaches 0
as r approaches infinity, but with oscillations that cross the zero axis.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WaveRBF(Double)` | Initializes a new instance of the `WaveRBF` class with a specified shape parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute()` | Computes the value of the Wave Radial Basis Function for a given radius. |
| `ComputeDerivative()` | Computes the derivative of the Wave RBF with respect to the radius. |
| `ComputeWidthDerivative()` | Computes the derivative of the Wave RBF with respect to the shape parameter epsilon. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_epsilon` | The shape parameter (epsilon) controlling the frequency of oscillations. |
| `_numOps` | The numeric operations provider for type T, used for mathematical calculations. |

