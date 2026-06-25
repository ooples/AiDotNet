---
title: "DataSetType"
description: "Represents the different types of datasets used in machine learning workflows."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Represents the different types of datasets used in machine learning workflows.

## For Beginners

In machine learning, we typically split our data into different sets for different purposes.

Think of it like learning to cook:

- Training data is like practicing recipes while learning (the model learns from this data)
- Validation data is like having someone taste your food and give feedback while you're still learning (helps tune your model)
- Testing data is like serving to customers who've never had your food before (final evaluation of your model)

This separation helps ensure that your model can generalize well to new, unseen data rather than just memorizing 
the examples it was trained on.

## Fields

| Field | Summary |
|:-----|:--------|
| `Testing` | The dataset used to evaluate the final model performance on unseen data. |
| `Training` | The dataset used to train the model by adjusting its parameters. |
| `Validation` | The dataset used during model development to tune hyperparameters and prevent overfitting. |

