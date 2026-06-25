---
title: "MaternRBF<T>"
description: "Implements a Matérn Radial Basis Function (RBF) that provides a flexible family of kernels."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RadialBasisFunctions`

Implements a Matérn Radial Basis Function (RBF) that provides a flexible family of kernels.

## For Beginners

A Radial Basis Function (RBF) is a special type of mathematical function
that depends only on the distance from a center point.

The Matérn RBF is like a "Swiss Army knife" of radial basis functions - it provides a whole family
of different shapes by adjusting a parameter called nu (?). This makes it very flexible for
different types of data.

This RBF has two main parameters:

- nu (?): Controls the smoothness of the function. Common values are 0.5, 1.5, and 2.5
- lengthScale (l): Controls how quickly the function decreases with distance

When nu = 0.5, the function decreases rapidly (exponentially) with distance.
When nu = 1.5, the function is smoother and decreases more gradually.
As nu increases, the function becomes even smoother, approaching a bell curve shape.

The Matérn RBF is popular in machine learning and statistics because you can adjust its
smoothness to match the characteristics of your data.

## How It Works

This class implements the Matérn family of radial basis functions, which are defined using modified Bessel functions
and provide a flexible set of kernels with varying degrees of smoothness. The Matérn RBF is defined as:
f(r) = [2^(1-?)/G(?)] × (v(2?)r/l)^? × K_?(v(2?)r/l)
where r is the radial distance, ? (nu) is a smoothness parameter, l is the length scale parameter,
G is the Gamma function, and K_? is the modified Bessel function of the second kind of order ?.

The Matérn function is commonly used in spatial statistics, machine learning, and geostatistics.
It generalizes many other RBFs; for example, when ? ? 8, it becomes the Gaussian RBF, and
when ? = 0.5, it becomes the exponential RBF. Special half-integer values of ? (0.5, 1.5, 2.5)
result in simpler forms that can be computed without Bessel functions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaternRBF(Double,Double)` | Initializes a new instance of the `MaternRBF` class with specified parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute()` | Computes the value of the Matérn Radial Basis Function for a given radius. |
| `ComputeDerivative()` | Computes the derivative of the Matérn RBF with respect to the radius. |
| `ComputeWidthDerivative()` | Computes the derivative of the Matérn RBF with respect to the length scale parameter. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lengthScale` | The length scale parameter controlling the width of the function. |
| `_nu` | The smoothness parameter (nu) controlling the differentiability of the function. |
| `_numOps` | The numeric operations provider for type T, used for mathematical calculations. |

