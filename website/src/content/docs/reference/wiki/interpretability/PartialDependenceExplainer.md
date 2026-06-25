---
title: "PartialDependenceExplainer<T>"
description: "Model-agnostic Partial Dependence Plot (PDP) explainer with Individual Conditional Expectation (ICE) curves."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Model-agnostic Partial Dependence Plot (PDP) explainer with Individual Conditional Expectation (ICE) curves.

## For Beginners

Partial Dependence Plots (PDPs) show how a feature affects predictions
on average, while holding all other features constant.

Imagine you want to know "How does Age affect loan approval probability?"

- PDP: Shows the average effect of Age across all applicants
- ICE: Shows individual curves for each applicant (revealing if the effect varies)

Key insights:

- Upward slope = feature increases predictions
- Downward slope = feature decreases predictions
- Flat line = feature has little effect
- If ICE curves are parallel = consistent effect for everyone
- If ICE curves cross = the effect depends on other features (interaction)

PDPs are great for understanding global feature effects, but can be misleading
when features are correlated. Use ALE (Accumulated Local Effects) for correlated features.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PartialDependenceExplainer(Func<Matrix<>,Vector<>>,Matrix<>,Int32,Boolean,String[])` | Initializes a new Partial Dependence explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeForFeature(Int32)` | Computes partial dependence for a single feature. |
| `ComputeForFeatures(Int32[])` | Computes partial dependence for multiple features. |
| `ComputeInteraction(Int32,Int32)` | Computes 2D partial dependence for feature interactions. |
| `ComputeSingleFeaturePD(Int32,PartialDependenceResult<>)` | Computes partial dependence for a single feature. |
| `ExplainGlobal(Matrix<>)` |  |

