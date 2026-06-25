---
title: "JaccardLossFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Jaccard Loss to evaluate model performance, particularly for segmentation and classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Jaccard Loss to evaluate model performance, particularly for segmentation and classification tasks.

## For Beginners

This calculator helps evaluate how well your model is performing on tasks where
you need to identify regions or categories, especially when you care about the overlap between
your predictions and the actual answers.

Jaccard Loss is based on the Jaccard Index (also called Intersection over Union or IoU),
which measures the similarity between two sets by comparing what they have in common
versus their combined elements.

How Jaccard Loss works:

- It calculates how much your prediction and the actual answer overlap
- It divides this overlap by the total area covered by both
- It then converts this similarity score into a loss (by subtracting from 1)

Think of it like this:
Imagine you and a friend are each drawing circles on a piece of paper:

- The Jaccard Index measures how much your circles overlap compared to their total area
- A score of 1 means perfect overlap (identical circles)
- A score of 0 means no overlap at all
- Jaccard Loss is simply 1 minus this score (so 0 is perfect, 1 is terrible)

Common applications include:

- Image segmentation (identifying regions in images)
- Object detection
- Multi-class classification
- Any task where you care about the overlap between predicted and actual regions

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `JaccardLossFitnessCalculator(DataSetType)` | Initializes a new instance of the JaccardLossFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Jaccard Loss fitness score for the given dataset. |

