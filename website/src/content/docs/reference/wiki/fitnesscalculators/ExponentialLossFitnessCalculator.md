---
title: "ExponentialLossFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Exponential Loss to evaluate model performance, particularly for classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Exponential Loss to evaluate model performance, particularly for classification tasks.

## For Beginners

This calculator helps you evaluate how well your model is performing on classification tasks,
with a special focus on heavily penalizing predictions that are both wrong and confident.

Exponential Loss works by:

- Giving a small penalty for minor mistakes (when your model is slightly unsure about the right answer)
- Giving a MUCH larger penalty for confident mistakes (when your model is very sure about a wrong answer)

Think of it like a teacher grading a test:

- If you answer "I'm not sure, but maybe A" and the answer is B, you get a small penalty
- If you answer "I'm 100% certain it's A!" and the answer is B, you get a huge penalty

Some common applications include:

- Fraud detection (where confidently missing a fraud case is very costly)
- Medical diagnosis (where confidently giving the wrong diagnosis could be dangerous)
- Any situation where being wrong AND confident is much worse than being unsure

Exponential Loss is used in algorithms like AdaBoost and can help your model focus on the examples
it's getting wrong, especially those where it's making confident mistakes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExponentialLossFitnessCalculator(DataSetType)` | Initializes a new instance of the ExponentialLossFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Exponential Loss between predicted and actual values. |

