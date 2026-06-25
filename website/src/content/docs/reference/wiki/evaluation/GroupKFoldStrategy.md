---
title: "GroupKFoldStrategy<T>"
description: "Group K-Fold: K-Fold that keeps related samples (groups) together in the same fold."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.CrossValidation`

Group K-Fold: K-Fold that keeps related samples (groups) together in the same fold.

## For Beginners

Group K-Fold ensures that samples from the same group are never
split between training and validation:

- Prevents data leakage when samples are related (e.g., multiple measurements from same patient)
- Groups could be: patients, subjects, time periods, locations, etc.
- Essential when independence assumption would be violated by standard K-Fold

## How It Works

**Example:** In medical data with multiple scans per patient, you want all scans from
a patient in the same fold. Otherwise, the model might "memorize" patient characteristics
from training and appear to perform well on validation (but fail on new patients).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GroupKFoldStrategy(Int32[],Nullable<Int32>)` | Initializes Group K-Fold cross-validation. |

