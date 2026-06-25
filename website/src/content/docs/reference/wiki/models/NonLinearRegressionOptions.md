---
title: "NonLinearRegressionOptions"
description: "Configuration options for nonlinear regression models, which capture complex, nonlinear relationships between input features and output variables using kernel functions and iterative optimization."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for nonlinear regression models, which capture complex, nonlinear relationships
between input features and output variables using kernel functions and iterative optimization.

## For Beginners

Nonlinear regression is a technique for finding patterns in data that don't follow straight lines.

Imagine trying to predict house prices:

- Linear regression might assume each additional square foot adds a fixed amount to the price
- Nonlinear regression can capture more realistic patterns, like:
- Diminishing returns (each extra square foot adds less value as houses get very large)
- Threshold effects (prices jump significantly once houses reach certain sizes)
- Interactions (extra bathrooms add more value in larger houses than in smaller ones)

This class provides settings that control:

- How precisely the model should fit the data
- How many attempts it should make to find the best fit
- What mathematical approach (kernel) to use for modeling curved relationships

Nonlinear regression is powerful because it can discover complex patterns that simpler
models miss, but it requires more careful configuration to work effectively. These options
help you control that balance between flexibility and reliability.

## How It Works

Nonlinear regression extends beyond the capabilities of linear models by allowing for curved or complex
relationships between variables. This class encapsulates the parameters that control how nonlinear
regression models are fitted to data, including convergence criteria, kernel selection, and kernel
hyperparameters. These models are particularly valuable when data exhibits patterns that cannot be
adequately represented by straight lines or hyperplanes, such as exponential growth, sinusoidal cycles,
or interactions between variables. The kernel approach transforms the input space to a higher-dimensional
feature space where complex relationships become more manageable.

## Properties

| Property | Summary |
|:-----|:--------|
| `Coef0` | Gets or sets the coef0 parameter used in Polynomial and Sigmoid kernels. |
| `Gamma` | Gets or sets the gamma parameter that controls the influence range in RBF, Polynomial, and Sigmoid kernels. |
| `KernelType` | Gets or sets the type of kernel function to use for transforming the input space. |
| `MaxIterations` | Gets or sets the maximum number of iterations allowed for the optimization algorithm. |
| `PolynomialDegree` | Gets or sets the degree of the polynomial when using the Polynomial kernel type. |
| `Tolerance` | Gets or sets the convergence tolerance that determines when the optimization algorithm should stop. |

