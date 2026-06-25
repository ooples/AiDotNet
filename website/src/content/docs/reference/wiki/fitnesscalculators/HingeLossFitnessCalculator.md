---
title: "HingeLossFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Hinge Loss to evaluate model performance, particularly for binary classification and support vector machines."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Hinge Loss to evaluate model performance, particularly for binary classification and support vector machines.

## For Beginners

This calculator helps evaluate how well your model is performing on classification tasks,
especially when you're trying to clearly separate two classes from each other.

Hinge Loss is commonly used with Support Vector Machines (SVMs), which are models that try to find
the best dividing line (or "hyperplane") between different classes of data.

How Hinge Loss works:

- It penalizes predictions that are both incorrect AND confident
- It doesn't care how correct your correct predictions are, only that they're on the right side of the boundary
- It creates a "margin" around the decision boundary and wants predictions to be clearly on one side or the other

Think of it like this:
Imagine you're trying to separate apples and oranges on a table with a stick.

- Hinge Loss doesn't just want the stick to separate them
- It wants all apples to be at least a certain distance from the stick on one side
- And all oranges to be at least a certain distance from the stick on the other side

This creates a "safety margin" that makes the model more robust.

Common applications include:

- Text classification (like spam detection)
- Image classification
- Any binary classification problem where you want a clear separation between classes

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HingeLossFitnessCalculator(DataSetType)` | Initializes a new instance of the HingeLossFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Hinge Loss fitness score for the given dataset. |

