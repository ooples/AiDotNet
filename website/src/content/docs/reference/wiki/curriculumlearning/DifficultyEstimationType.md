---
title: "DifficultyEstimationType"
description: "Types of difficulty estimation methods."
section: "API Reference"
---

`Enums` · `AiDotNet.CurriculumLearning.Interfaces`

Types of difficulty estimation methods.

## Fields

| Field | Summary |
|:-----|:--------|
| `ComplexityBased` | Uses sample complexity metrics (e.g., input magnitude). |
| `ConfidenceBased` | Uses model confidence/margin as difficulty measure. |
| `Ensemble` | Combines multiple estimators. |
| `EnsembleBased` | Uses prediction variance across model ensemble. |
| `ExpertDefined` | Uses domain expert-defined difficulty scores. |
| `LossBased` | Uses training loss as difficulty measure. |
| `TransferBased` | Uses gap between simple and complex model performance. |

