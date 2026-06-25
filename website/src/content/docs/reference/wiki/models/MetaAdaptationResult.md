---
title: "MetaAdaptationResult<T>"
description: "Results from adapting a meta-learner to a single task."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Results from adapting a meta-learner to a single task.

## For Beginners

Meta-learning has two phases for each task:

1. **Adaptation (Inner Loop):** The model quickly learns from the support set (a few examples)
2. **Evaluation:** We test on the query set (held-out examples) to see if it really learned

This result tracks:

- **Support metrics:** How well the model fits the training examples (should be very good)
- **Query metrics:** How well it generalizes (the real test!)
- **Adaptation details:** How many steps, how long it took, etc.

For example, in 5-way 1-shot learning:

- Support set: 5 examples (1 per class) - model adapts on these
- Query set: 15 examples (3 per class) - model is evaluated on these
- Good meta-learning: High query accuracy even with tiny support set

## How It Works

This class captures detailed metrics from the inner loop adaptation process for a single task.
It tracks both support set performance (where the model adapts) and query set performance
(where we measure true generalization), along with adaptation details.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaAdaptationResult(,,,,Int32,Double,List<>,Dictionary<String,>)` | Initializes a new instance with adaptation metrics. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets the number of gradient steps taken during adaptation. |
| `AdaptationTimeMs` | Gets the time taken for adaptation in milliseconds. |
| `AdditionalMetrics` | Gets algorithm-specific metrics for this adaptation. |
| `PerStepLosses` | Gets the loss values recorded at each adaptation step (optional). |
| `QueryAccuracy` | Gets the accuracy on the query set after adaptation. |
| `QueryLoss` | Gets the loss on the query set after adaptation. |
| `SupportAccuracy` | Gets the accuracy on the support set after adaptation. |
| `SupportLoss` | Gets the loss on the support set after adaptation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DidConverge(Double)` | Checks if the adaptation converged based on loss reduction. |
| `GenerateReport` | Generates a formatted summary of adaptation results. |

