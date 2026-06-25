---
title: "MissingFeatureStrategy"
description: "Specifies how to handle missing feature blocks when not all parties have data for all entities."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies how to handle missing feature blocks when not all parties have data for all entities.

## For Beginners

In vertical FL, different parties may not have data for all entities.
For example, Hospital A has records for patients 1-1000 and Hospital B has records for patients
500-1500. For patients 1-499 (only in A) and 1001-1500 (only in B), the other party's features
are "missing". This enum controls how those missing features are filled in during training.

## Fields

| Field | Summary |
|:-----|:--------|
| `Learned` | Use a learned imputation model that predicts missing features from available features. |
| `Mean` | Replace missing features with the column-wise mean of available data. |
| `Skip` | Skip entities with missing features entirely. |
| `Zero` | Replace missing features with zeros. |

