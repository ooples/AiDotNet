---
title: "WeightFunctionHelper<T>"
description: "Provides methods for calculating weights used in robust regression techniques."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides methods for calculating weights used in robust regression techniques.

## How It Works

**For Beginners:** In standard regression, all data points are treated equally. However, in real-world data,
some points may be outliers (unusual values that don't follow the general pattern). Robust regression
techniques handle these outliers by assigning different "weights" to different data points.

Think of weights like importance scores:

- Normal data points get high weights (close to 1), meaning they have full influence on the model
- Outliers get low weights (close to 0), reducing their influence on the model

This helper class calculates these weights using different mathematical formulas (Huber, Bisquare, Andrews)
that determine how aggressively to downweight outliers.

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateAndrewsWeights(Vector<>,Double)` | Calculates weights using Andrews' sine wave method. |
| `CalculateBisquareWeights(Vector<>,Double)` | Calculates weights using Tukey's bisquare (biweight) method. |
| `CalculateHuberWeights(Vector<>,Double)` | Calculates weights using Huber's method. |
| `CalculateWeights(Vector<>,WeightFunction,Double)` | Calculates weights for data points based on their residuals using the specified weight function. |

