---
title: "NelderMeadOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the Nelder-Mead optimization algorithm, a derivative-free method for finding the minimum of an objective function in a multidimensional space."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Nelder-Mead optimization algorithm, a derivative-free method
for finding the minimum of an objective function in a multidimensional space.

## For Beginners

The Nelder-Mead algorithm is a clever way to find the minimum value of a function
without needing to calculate derivatives (which can be complicated or impossible for some problems).

Imagine you're trying to find the lowest point in a hilly landscape:

- Instead of following the steepest downhill path (as gradient-based methods do)
- Nelder-Mead places a "flexible shape" on the landscape (like a triangle in 2D space)
- It then moves and reshapes this triangle, always trying to "slide" toward lower ground

The algorithm works through a series of simple operations:

- Reflection: Try moving away from the highest point
- Expansion: If that works well, try moving even further
- Contraction: If reflection doesn't work, try moving a shorter distance
- Shrinking: If all else fails, make the triangle smaller and try again

This class allows you to configure how aggressively or cautiously these operations are performed,
which affects how quickly and reliably the algorithm can find the optimal solution.

## How It Works

The Nelder-Mead method (also known as the downhill simplex method or amoeba method) is a numerical
optimization technique that does not require derivative information. It works by constructing a simplex
(a generalization of a triangle to higher dimensions) and systematically replacing the worst vertex
of this simplex with a new point through a series of reflection, expansion, contraction, and shrinking
operations. This approach is particularly useful for problems where the objective function is non-differentiable,
is not known in closed form, or when gradient calculations are computationally expensive or unstable.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationRate` | Gets or sets the rate at which parameters are adjusted when using adaptive parameters. |
| `InitialAlpha` | Gets or sets the initial reflection coefficient, which controls how far to reflect the simplex away from the worst point. |
| `InitialBeta` | Gets or sets the initial contraction coefficient, which controls how far to contract the simplex toward the centroid. |
| `InitialDelta` | Gets or sets the initial shrink coefficient, which controls how much to shrink the entire simplex when other operations fail. |
| `InitialGamma` | Gets or sets the initial expansion coefficient, which controls how far to expand the simplex in a promising direction. |
| `MaxAlpha` | Gets or sets the maximum allowed value for the reflection coefficient when using adaptive parameters. |
| `MaxBeta` | Gets or sets the maximum allowed value for the contraction coefficient when using adaptive parameters. |
| `MaxDelta` | Gets or sets the maximum allowed value for the shrink coefficient when using adaptive parameters. |
| `MaxGamma` | Gets or sets the maximum allowed value for the expansion coefficient when using adaptive parameters. |
| `MinAlpha` | Gets or sets the minimum allowed value for the reflection coefficient when using adaptive parameters. |
| `MinBeta` | Gets or sets the minimum allowed value for the contraction coefficient when using adaptive parameters. |
| `MinDelta` | Gets or sets the minimum allowed value for the shrink coefficient when using adaptive parameters. |
| `MinGamma` | Gets or sets the minimum allowed value for the expansion coefficient when using adaptive parameters. |
| `UseAdaptiveParameters` | Gets or sets whether to adaptively adjust the algorithm parameters based on their effectiveness. |

