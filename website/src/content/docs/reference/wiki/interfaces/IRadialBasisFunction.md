---
title: "IRadialBasisFunction<T>"
description: "Defines a radial basis function (RBF) that measures similarity based on distance."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines a radial basis function (RBF) that measures similarity based on distance.

## How It Works

Radial basis functions are mathematical functions whose value depends only on the distance
from a central point. They are commonly used in machine learning for creating complex models
from simpler building blocks.

**For Beginners:** Think of a radial basis function as a "similarity detector" that works like this:

- It measures how similar or close two points are to each other
- The closer two points are, the higher the output value
- The function creates a smooth "hill" or "bump" shape centered at a specific point
- As you move away from the center, the function's value decreases

Common examples include the Gaussian (bell curve) function and the multiquadric function.
These are used in many AI applications like:

- Function approximation (finding patterns in data)
- Classification (sorting data into categories)
- Time series prediction (forecasting future values)

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute()` | Calculates the value of the radial basis function at a given distance. |
| `ComputeDerivative()` | Calculates the derivative (rate of change) of the radial basis function with respect to distance. |
| `ComputeWidthDerivative()` | Calculates the derivative of the radial basis function with respect to its width parameter. |

