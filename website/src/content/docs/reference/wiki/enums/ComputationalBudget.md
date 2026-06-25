---
title: "ComputationalBudget"
description: "Describes the computational budget available for model training and search."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Describes the computational budget available for model training and search.

## For Beginners

This tells AutoML how much computing time and resources you're willing to spend.
Lower budgets prefer simpler, faster models. Higher budgets allow expensive ensembles and deep learning.

## Fields

| Field | Summary |
|:-----|:--------|
| `High` | High budget — can afford expensive models and extensive hyperparameter search. |
| `Low` | Low budget — prefer fast, simple models (linear, small trees). |
| `Moderate` | Moderate budget — typical training resources (ensembles, medium networks). |

