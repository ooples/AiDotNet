---
title: "ContrastiveLossFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Contrastive Loss to evaluate model performance for similarity learning tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Contrastive Loss to evaluate model performance for similarity learning tasks.

## For Beginners

This calculator helps you evaluate how well your model is learning to determine if two items are similar or different.

Contrastive Loss is used in "similarity learning" - a type of machine learning where the goal is to learn 
which items are similar and which are different. Some common applications include:

- Face recognition (are these two photos of the same person?)
- Signature verification (did the same person sign both documents?)
- Product recommendations (finding similar products)
- Duplicate detection (finding duplicate documents or images)

The loss function works by:

- Pulling similar items closer together in the feature space
- Pushing dissimilar items farther apart, up to a certain distance (called the "margin")

Think of it like organizing a classroom: you want students working on the same project to sit close together,
while students working on different projects should sit at least a certain distance apart.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContrastiveLossFitnessCalculator(,DataSetType)` | Initializes a new instance of the ContrastiveLossFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateSimilarityLabel(,)` | Determines if two samples should be considered similar or different. |
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Contrastive Loss between predicted and actual values. |
| `SplitOutputs(Vector<>)` | Splits a vector into two equal parts, representing the first and second items in each pair. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_margin` | The margin value that defines the minimum distance between dissimilar pairs. |

