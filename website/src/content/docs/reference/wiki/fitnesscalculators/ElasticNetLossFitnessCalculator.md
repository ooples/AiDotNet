---
title: "ElasticNetLossFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Elastic Net Loss to evaluate model performance while encouraging simpler models through regularization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Elastic Net Loss to evaluate model performance while encouraging simpler models through regularization.

## For Beginners

This calculator helps you evaluate how well your model is performing while also encouraging it to be simpler.

Elastic Net Loss combines two approaches to keep your model from becoming too complex:

1. It measures how well your predictions match the actual values (like other loss functions)
2. It adds penalties for having too many or too large parameters in your model

Think of it like building a bridge:

- You want the bridge to be strong enough to do its job (make good predictions)
- But you also want to use as few materials as possible (keep the model simple)
- Elastic Net helps you find this balance

Some common applications include:

- Financial predictions where you want to identify only the most important factors
- Medical models where you need to know which few symptoms are most predictive
- Any situation where you have many potential input features but want to use only the most important ones

Elastic Net is particularly useful when you have many input features that might be related to each other,
as it helps select the most important ones while reducing the impact of less important or redundant features.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ElasticNetLossFitnessCalculator(,,DataSetType)` | Initializes a new instance of the ElasticNetLossFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Elastic Net Loss between predicted and actual values, including regularization penalties. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | The strength of the regularization penalty applied to the model. |
| `_l1Ratio` | The ratio that determines the mix between L1 (absolute value) and L2 (squared value) regularization. |

