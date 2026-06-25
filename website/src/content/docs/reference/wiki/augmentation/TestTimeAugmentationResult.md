---
title: "TestTimeAugmentationResult<TOutput>"
description: "Contains the result of a Test-Time Augmentation prediction, including individual and aggregated predictions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation`

Contains the result of a Test-Time Augmentation prediction, including individual and aggregated predictions.

## For Beginners

This class gives you both:

1. The final combined prediction (what you usually want)
2. All the individual predictions (for debugging or analysis)

You also get uncertainty information - if all 5 predictions were similar, you can be
confident in the result. If they varied wildly, you might want to be more cautious.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TestTimeAugmentationResult(,IReadOnlyList<>,Nullable<Double>,Nullable<Double>)` | Creates a new Test-Time Augmentation result. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AggregatedPrediction` | Gets the final combined prediction after aggregating all augmented predictions. |
| `Confidence` | Gets the confidence score of the aggregated prediction (if available). |
| `IndividualPredictions` | Gets all the individual predictions from each augmented version. |
| `StandardDeviation` | Gets the standard deviation of predictions, measuring uncertainty. |

