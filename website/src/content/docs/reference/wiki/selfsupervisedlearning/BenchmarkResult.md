---
title: "BenchmarkResult<T>"
description: "Result from a single benchmark evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning.Evaluation`

Result from a single benchmark evaluation.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdditionalMetrics` | Additional metrics (e.g., different k values for k-NN). |
| `Protocol` | The evaluation protocol used. |
| `SamplePercentage` | Percentage of training data used (for few-shot). |
| `Top1Accuracy` | Top-1 classification accuracy. |
| `Top5Accuracy` | Top-5 classification accuracy. |
| `TrainingHistory` | Training accuracy history. |
| `ValidationHistory` | Validation accuracy history. |

