---
title: "PolynomialMean<T>"
description: "Implements a polynomial mean function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Implements a polynomial mean function.

## For Beginners

The polynomial mean function returns a polynomial of the input features.

For 1D input: m(x) = a_0 + a_1×x + a_2×x² + ... + a_d×x^d

Why use polynomial mean?

- When you expect a curved trend in your data
- For extrapolation beyond training data with a known trend shape
- When physics/domain knowledge suggests polynomial behavior

Warning: High-degree polynomials can cause problems:

- Extrapolation becomes unstable
- May overfit to noise
- Consider degree 2 or 3 as maximum for most applications

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PolynomialMean(Double[])` | Initializes a polynomial mean function for 1D input. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Coefficients` | Gets a copy of the coefficients. |
| `Degree` | Gets the polynomial degree. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Matrix<>)` | Computes the polynomial mean values at multiple input points. |
| `Evaluate(Vector<>)` | Computes the polynomial mean value at the given input point. |

