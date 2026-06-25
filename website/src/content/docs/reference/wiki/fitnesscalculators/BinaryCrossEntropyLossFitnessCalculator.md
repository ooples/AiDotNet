---
title: "BinaryCrossEntropyLossFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Binary Cross-Entropy Loss to evaluate model performance for binary classification problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Binary Cross-Entropy Loss to evaluate model performance for binary classification problems.

## For Beginners

This calculator helps you evaluate how well your model is performing on binary classification tasks
(problems where you're predicting one of two possible outcomes, like "yes/no" or "spam/not spam").

Binary Cross-Entropy Loss measures the difference between your model's predicted probabilities and the actual
outcomes. Here's how to understand it:

- It's specifically designed for binary classification problems (where the answer is one of two possibilities)
- Lower values are better (0 would be a perfect model)
- It heavily penalizes confident but wrong predictions (e.g., if your model is 99% sure something is spam when it's not)
- It's commonly used in neural networks and logistic regression

Think of it like a scoring system that gives your model a harsh penalty when it's very confident but wrong,
and a small penalty when it's uncertain. This encourages your model to be confident only when it has good reason to be.

Unlike accuracy (which just counts right vs. wrong), Binary Cross-Entropy Loss takes into account how confident
your model was in its predictions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BinaryCrossEntropyLossFitnessCalculator(DataSetType)` | Initializes a new instance of the BinaryCrossEntropyLossFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Binary Cross-Entropy Loss between predicted and actual values. |

