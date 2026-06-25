---
title: "CategoricalCrossEntropyLossFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Categorical Cross-Entropy Loss to evaluate model performance for multi-class classification problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Categorical Cross-Entropy Loss to evaluate model performance for multi-class classification problems.

## For Beginners

This calculator helps you evaluate how well your model is performing on multi-class classification tasks
(problems where you're predicting one of several possible categories, like "dog/cat/bird" or "red/green/blue/yellow").

Categorical Cross-Entropy Loss measures how well your model's predicted probabilities match the actual categories.
Here's how to understand it:

- It's designed for problems with multiple possible categories (3 or more classes)
- Lower values are better (0 would be a perfect model)
- It heavily penalizes confident but wrong predictions (e.g., if your model is 95% sure an image is a dog when it's actually a cat)
- It's commonly used in neural networks for image classification, text categorization, and other multi-class problems

While Binary Cross-Entropy Loss works with two categories (like yes/no questions), Categorical Cross-Entropy Loss
handles multiple categories (like multiple-choice questions).

Think of it like a teacher grading a multiple-choice test: your model gets more points when it's confident about
the right answer and loses points when it's confident about the wrong answer.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CategoricalCrossEntropyLossFitnessCalculator(DataSetType)` | Initializes a new instance of the CategoricalCrossEntropyLossFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Categorical Cross-Entropy Loss between predicted and actual values. |

