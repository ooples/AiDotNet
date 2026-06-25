---
title: "FeatureInteractionExplainer<T>"
description: "Model-agnostic Feature Interaction detector using Friedman's H-statistic."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Model-agnostic Feature Interaction detector using Friedman's H-statistic.

## For Beginners

The H-statistic measures how much features interact with each other.

What is a feature interaction?

- When the effect of one feature depends on the value of another feature
- Example: "Education increases salary, but more so for people with more Experience"
- This means Education and Experience interact

How to interpret H-statistic values:

- H = 0: No interaction (features act independently)
- H = 1: Pure interaction (entire effect comes from interaction)
- H between 0 and 1: Partial interaction

Typical thresholds:

- H < 0.05: Negligible interaction
- H 0.05-0.20: Weak interaction
- H 0.20-0.50: Moderate interaction
- H > 0.50: Strong interaction

This implementation computes:

1. Pairwise H-statistics (between two specific features)
2. Overall H-statistic (one feature vs all others)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FeatureInteractionExplainer(Func<Matrix<>,Vector<>>,Matrix<>,Int32,String[])` | Initializes a new Feature Interaction explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeOverallHStatistic(Int32)` | Computes the overall H-statistic for a single feature vs all others. |
| `ComputePairwiseHStatistic(Int32,Int32)` | Computes the pairwise H-statistic between two features. |
| `EnsurePD2DCached(Int32,Int32)` | Ensures 2D partial dependence is cached for a feature pair. |
| `EnsurePDCached(Int32)` | Ensures 1D partial dependence is cached for a feature. |
| `ExplainGlobal(Matrix<>)` |  |
| `GetTopInteractions(Int32)` | Gets the top interacting feature pairs. |

