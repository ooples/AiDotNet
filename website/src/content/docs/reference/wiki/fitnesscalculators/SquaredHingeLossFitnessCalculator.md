---
title: "SquaredHingeLossFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Squared Hinge Loss to evaluate model performance, particularly for binary classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Squared Hinge Loss to evaluate model performance, particularly for binary classification tasks.

## For Beginners

This calculator helps evaluate how well your model is performing on classification tasks,
especially when you're trying to decide if something belongs to a category or not (binary classification).

Squared Hinge Loss is a variation of Hinge Loss that puts even more emphasis on getting predictions right.

How Squared Hinge Loss works:

- It expects predictions to be either -1 (for negative class) or +1 (for positive class)
- It wants predictions to be not just correct, but confident (with a margin of safety)
- It penalizes incorrect or uncertain predictions by squaring the error, which makes larger errors much more significant

Think of it like this:
Imagine you're a teacher grading true/false questions:

- Students get full credit for being confidently correct
- Students lose points for being wrong or uncertain
- The more confidently wrong they are, the exponentially more points they lose

Compared to regular Hinge Loss:

- Regular Hinge Loss increases penalties linearly for wrong predictions
- Squared Hinge Loss increases penalties quadratically (much faster) for wrong predictions
- This makes Squared Hinge Loss more sensitive to outliers and large errors

Common applications include:

- Email spam detection
- Sentiment analysis (positive/negative)
- Medical diagnosis (presence/absence of a condition)
- Any binary classification problem where you want to strongly penalize misclassifications

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SquaredHingeLossFitnessCalculator(DataSetType)` | Initializes a new instance of the SquaredHingeLossFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Squared Hinge Loss fitness score for the given dataset. |

