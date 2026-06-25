---
title: "WeightedCrossEntropyLossFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Weighted Cross Entropy Loss to evaluate model performance, particularly for classification problems with imbalanced classes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Weighted Cross Entropy Loss to evaluate model performance, particularly for classification problems with imbalanced classes.

## For Beginners

This calculator helps evaluate how well your model is performing on classification tasks,
especially when some classes are more important than others or appear less frequently in your data.

Cross Entropy Loss is a common way to measure how well a classification model is performing.
The "weighted" part means we can give more importance to certain classes.

Think of it like grading a test:

- Regular Cross Entropy treats all questions equally
- Weighted Cross Entropy lets you assign more points to harder or more important questions

This is particularly useful when:

- Some classes appear much less frequently than others (like rare diseases in medical diagnosis)
- Some mistakes are more costly than others (like falsely classifying a tumor as benign)
- You want to focus the model's attention on specific classes

By providing weights, you can:

- Increase the penalty for misclassifying minority classes
- Balance the learning process when your data is imbalanced
- Prioritize certain types of predictions based on their importance

Common applications include:

- Medical diagnosis (where false negatives might be dangerous)
- Fraud detection (where fraud is rare but important to catch)
- Any classification problem with imbalanced classes

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WeightedCrossEntropyLossFitnessCalculator(Vector<>,DataSetType)` | Initializes a new instance of the WeightedCrossEntropyLossFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Weighted Cross Entropy Loss fitness score for the given dataset. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_weights` | The weights to apply to each class when calculating the cross entropy loss. |

