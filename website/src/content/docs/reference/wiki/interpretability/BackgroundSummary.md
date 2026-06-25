---
title: "BackgroundSummary<T>"
description: "Represents summarized background data for interpretability methods."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Helpers`

Represents summarized background data for interpretability methods.

## For Beginners

This class holds the summarized background data
along with weights for each sample.

The weights indicate how "representative" each sample is:

- Higher weight = represents more of the original data
- Weights sum to 1.0

**Example:** If cluster A had 1000 samples and cluster B had 100 samples,
the center of cluster A would have ~10x the weight of cluster B.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BackgroundSummary(Matrix<>,Vector<>)` | Initializes a new background summary. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Data` | Gets the summarized background data (rows = samples). |
| `NumFeatures` | Gets the number of features. |
| `NumSamples` | Gets the number of background samples. |
| `Weights` | Gets the weight for each background sample. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExpectedPrediction(Func<Vector<>,Vector<>>)` | Gets the expected prediction using the background data. |
| `GetWeightedMean` | Gets the weighted mean of each feature. |
| `ToString` | Returns a human-readable summary. |

