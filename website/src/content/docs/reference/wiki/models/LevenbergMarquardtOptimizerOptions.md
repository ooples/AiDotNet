---
title: "LevenbergMarquardtOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the Levenberg-Marquardt optimization algorithm, which is used for non-linear least squares optimization in machine learning and AI models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Levenberg-Marquardt optimization algorithm, which is used
for non-linear least squares optimization in machine learning and AI models.

## For Beginners

The Levenberg-Marquardt algorithm is a powerful technique for training
AI models that need to make accurate predictions. It works by repeatedly adjusting the model's
internal settings (parameters) to reduce prediction errors.

Think of it like tuning a musical instrument:

- You listen to how "off" the sound is (the error)
- You make small adjustments to the tuning pegs
- You check if the sound improved or got worse
- You keep adjusting until the instrument sounds right

The "damping factor" controls how boldly or cautiously the algorithm makes adjustments:

- Higher damping = smaller, more careful adjustments (slower but more stable)
- Lower damping = larger, more aggressive adjustments (faster but potentially unstable)

The algorithm automatically adjusts this damping factor as it progresses, becoming more
aggressive when things are going well and more cautious when improvements are hard to find.
This class allows you to configure how this damping behaves during training.

## How It Works

The Levenberg-Marquardt algorithm combines the Gauss-Newton method and gradient descent
to efficiently solve non-linear least squares problems. It adaptively switches between the two
approaches depending on how well the optimization is progressing, offering both stability and speed.
The damping factor controls this adaptation by determining whether the algorithm behaves more like
gradient descent (higher damping) or more like Gauss-Newton (lower damping).

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for gradient computation. |
| `CustomDecomposition` | Gets or sets a custom matrix decomposition method for solving the linear system in each Levenberg-Marquardt iteration. |
| `DampingFactorDecreaseFactor` | Gets or sets the factor by which the damping factor is decreased when an iteration successfully improves the solution. |
| `DampingFactorIncreaseFactor` | Gets or sets the factor by which the damping factor is increased when an iteration fails to improve the solution. |
| `InitialDampingFactor` | Gets or sets the starting value for the damping factor used in the algorithm. |
| `MaxDampingFactor` | Gets or sets the maximum allowed value for the damping factor. |
| `MinDampingFactor` | Gets or sets the minimum allowed value for the damping factor. |
| `UseAdaptiveDampingFactor` | Gets or sets whether the damping factor should be adaptively updated based on the success or failure of each iteration. |

