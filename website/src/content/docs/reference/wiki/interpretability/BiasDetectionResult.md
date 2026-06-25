---
title: "BiasDetectionResult<T>"
description: "Represents the results of a bias detection analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability`

Represents the results of a bias detection analysis.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BiasDetectionResult` | Initializes a new instance of the BiasDetectionResult class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DisparateImpactRatio` | Gets or sets the disparate impact ratio (min rate / max rate). |
| `EqualOpportunityDifference` | Gets or sets the equal opportunity difference (max TPR - min TPR). |
| `GroupFalsePositiveRates` | Gets or sets the False Positive Rates for each group. |
| `GroupPositiveRates` | Gets or sets the positive prediction rates for each group. |
| `GroupPrecisions` | Gets or sets the Precision values for each group. |
| `GroupSizes` | Gets or sets the sizes of each group. |
| `GroupTruePositiveRates` | Gets or sets the True Positive Rates for each group. |
| `HasBias` | Gets or sets whether bias was detected. |
| `Message` | Gets or sets the message describing the bias detection results. |
| `StatisticalParityDifference` | Gets or sets the statistical parity difference (max rate - min rate). |

