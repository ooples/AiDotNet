---
title: "SelfPaceRegularizer"
description: "Type of self-pace regularizer for sample weighting."
section: "API Reference"
---

`Enums` · `AiDotNet.CurriculumLearning.Schedulers`

Type of self-pace regularizer for sample weighting.

## Fields

| Field | Summary |
|:-----|:--------|
| `Hard` | Binary selection: include if loss < lambda. |
| `Linear` | Linear soft weighting: weight = max(0, 1 - loss/lambda). |
| `Logarithmic` | Logarithmic weighting for smoother transitions. |
| `Mixture` | Mixture of hard and linear for balanced selection. |

