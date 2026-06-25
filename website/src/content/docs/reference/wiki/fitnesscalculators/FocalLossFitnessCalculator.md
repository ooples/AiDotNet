---
title: "FocalLossFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Focal Loss to evaluate model performance, particularly for imbalanced classification problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Focal Loss to evaluate model performance, particularly for imbalanced classification problems.

## For Beginners

This calculator helps evaluate how well your model is performing on classification tasks,
especially when some classes appear much more frequently than others in your data.

Focal Loss is designed to solve a common problem in AI:
When one class is very common (like "normal emails") and another is rare (like "spam emails"),
models tend to focus too much on the common class and perform poorly on the rare class.

Focal Loss works by:

- Giving more importance to the difficult, misclassified examples
- Reducing the importance of the easy, well-classified examples

Think of it like a teacher who spends more time helping students with difficult problems
and less time on problems the students already understand well.

The two main parameters that control Focal Loss are:

- gamma: Controls how much to focus on hard-to-classify examples (higher values = more focus)
- alpha: Helps balance between different classes (adjusts for class imbalance)

Common applications include:

- Object detection in images (where most of the image is background)
- Medical diagnosis of rare conditions
- Fraud detection (where most transactions are legitimate)
- Any situation with imbalanced classes

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FocalLossFitnessCalculator(,,DataSetType)` | Initializes a new instance of the FocalLossFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Focal Loss fitness score for the given dataset. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gamma` | The focusing parameter that controls how much to down-weight easy examples. |

