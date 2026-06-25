---
title: "EnsembleAggregationMode"
description: "Defines how ensemble predictions are aggregated."
section: "API Reference"
---

`Enums` · `AiDotNet.KnowledgeDistillation.Teachers`

Defines how ensemble predictions are aggregated.

## Fields

| Field | Summary |
|:-----|:--------|
| `GeometricMean` | Geometric mean of teacher logits (for multiplicative ensembles). |
| `Maximum` | Element-wise maximum (for pessimistic ensembles). |
| `Median` | Element-wise median (robust to outliers). |
| `WeightedAverage` | Weighted average of teacher logits (most common). |

