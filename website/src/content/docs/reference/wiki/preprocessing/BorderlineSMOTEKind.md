---
title: "BorderlineSMOTEKind"
description: "Specifies the variant of Borderline-SMOTE to use."
section: "API Reference"
---

`Enums` · `AiDotNet.Preprocessing.ImbalancedLearning`

Specifies the variant of Borderline-SMOTE to use.

## For Beginners

- Kind1: Safer, only creates samples between minority class points
- Kind2: More diverse, can create samples between minority and majority points

## Fields

| Field | Summary |
|:-----|:--------|
| `Kind1` | Interpolates only between borderline samples and their minority class neighbors. |
| `Kind2` | Interpolates between borderline samples and any neighbor (including majority class). |

