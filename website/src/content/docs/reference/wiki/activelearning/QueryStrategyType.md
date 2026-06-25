---
title: "QueryStrategyType"
description: "Types of query strategies for active learning."
section: "API Reference"
---

`Enums` · `AiDotNet.ActiveLearning.Config`

Types of query strategies for active learning.

## Fields

| Field | Summary |
|:-----|:--------|
| `BADGE` | BADGE: Batch Active learning by Diverse Gradient Embeddings. |
| `BALD` | Bayesian Active Learning by Disagreement - uses MC Dropout. |
| `BatchBALD` | Batch variant of BALD that considers joint information. |
| `CoreSet` | Select samples that are representative of the unlabeled pool. |
| `Diversity` | Select diverse samples in feature space. |
| `Entropy` | Select samples with highest prediction entropy. |
| `ExpectedErrorReduction` | Select samples that would reduce expected error most. |
| `ExpectedModelChange` | Select samples that would cause largest gradient update. |
| `InformationDensity` | Combine uncertainty with density in feature space. |
| `LearningLoss` | Learning Loss - learns to predict loss for query selection. |
| `LeastConfidence` | Select samples with lowest maximum predicted probability. |
| `Margin` | Select samples with smallest margin between top predictions. |
| `QBC` | Query By Committee - uses ensemble disagreement. |
| `Random` | Random sampling baseline. |
| `UncertaintySampling` | Query samples where the model is most uncertain. |
| `VarianceReduction` | Select samples that would reduce prediction variance most. |

