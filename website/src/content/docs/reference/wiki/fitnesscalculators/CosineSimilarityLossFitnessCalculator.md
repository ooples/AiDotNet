---
title: "CosineSimilarityLossFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Cosine Similarity Loss to evaluate model performance for tasks where the direction of vectors matters more than their magnitude."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Cosine Similarity Loss to evaluate model performance for tasks where the direction of vectors matters more than their magnitude.

## For Beginners

This calculator helps you evaluate how well your model is learning to predict vectors that point in the same direction.

Cosine Similarity measures the angle between two vectors (sets of numbers), ignoring their length or magnitude.
It's like comparing the direction two people are facing, without caring how far away they are standing.

Some common applications include:

- Text similarity (comparing document topics)
- Recommendation systems (finding similar products or content)
- Image retrieval (finding images with similar content)
- Natural language processing (comparing word meanings)

The loss ranges from:

- 0 (best): Vectors point in exactly the same direction
- 2 (worst): Vectors point in exactly opposite directions

For example, if your model is trying to learn word meanings, cosine similarity would help determine if
two words are related in meaning (pointing in similar directions) regardless of how common or rare they are.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CosineSimilarityLossFitnessCalculator(DataSetType)` | Initializes a new instance of the CosineSimilarityLossFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Cosine Similarity Loss between predicted and actual values. |

