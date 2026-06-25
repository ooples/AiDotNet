---
title: "AccumulatedLocalEffectsExplainer<T>"
description: "Model-agnostic Accumulated Local Effects (ALE) explainer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Model-agnostic Accumulated Local Effects (ALE) explainer.

## For Beginners

ALE plots are an improved version of Partial Dependence Plots (PDP).
They show how a feature affects predictions, but handle correlated features much better.

The key difference from PDP:

- PDP: Averages over ALL data points (can create unrealistic combinations)
- ALE: Only looks at ACTUAL data points in each interval (realistic combinations)

Example: If you have Age and Years of Experience (highly correlated),
PDP might evaluate "Age=25 with 40 years experience" (impossible!).
ALE avoids this by only looking at real data within each age range.

How ALE works:

1. Divide the feature into intervals (bins)
2. For each interval, compute the effect as the average difference in predictions

when moving from the left edge to the right edge of the interval

3. Accumulate these effects to get the final ALE curve
4. Center the curve so the average effect is zero

When to use ALE vs PDP:

- Use ALE when features are correlated (most real-world cases)
- Use PDP when features are independent

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AccumulatedLocalEffectsExplainer(Func<Matrix<>,Vector<>>,Matrix<>,Int32,String[])` | Initializes a new Accumulated Local Effects explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Accumulate2D([0:,0:],Int32)` | Accumulates 2D local effects. |
| `ComputeForFeature(Int32)` | Computes ALE for a single feature. |
| `ComputeForFeatures(Int32[])` | Computes ALE for multiple features. |
| `ComputeInteraction(Int32,Int32)` | Computes 2D ALE for feature interactions. |
| `ComputeQuantileBounds(Int32,Int32)` | Computes quantile-based bounds for a feature. |
| `ComputeSingleFeatureALE(Int32,ALEResult<>)` | Computes ALE for a single feature. |
| `CreateSingleRowMatrix(Int32,Int32,Double)` | Creates a single-row matrix with a modified feature value. |
| `ExplainGlobal(Matrix<>)` |  |
| `FindInterval(Double,[])` | Finds the interval index for a given value. |
| `PredictWithModifiedFeatures(Int32,Int32,Double,Int32,Double)` | Makes a prediction with two modified feature values. |

