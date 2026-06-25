---
title: "DatasetSplit"
description: "Represents the type of dataset split."
section: "API Reference"
---

`Enums` · `AiDotNet.MetaLearning.Data`

Represents the type of dataset split.

## For Beginners

In machine learning, we typically split our data into three parts:

- Train: Used during meta-training to learn how to learn
- Validation: Used to tune hyperparameters without overfitting
- Test: Used for final evaluation to see how well the model generalizes

In meta-learning, each split contains different classes to ensure the model learns to
generalize to completely new tasks, not just new examples of seen classes.

## Fields

| Field | Summary |
|:-----|:--------|
| `Test` | Test split used for final evaluation. |
| `Train` | Training split used for meta-training. |
| `Validation` | Validation split used for hyperparameter tuning and early stopping. |

