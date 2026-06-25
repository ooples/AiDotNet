---
title: "TripletLossFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Triplet Loss to evaluate model performance, particularly for similarity learning and embedding tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Triplet Loss to evaluate model performance, particularly for similarity learning and embedding tasks.

## For Beginners

This calculator helps train models that learn to recognize similarities between items.

Triplet Loss works with three examples at a time:

- An "anchor" (the reference item)
- A "positive" (an item similar to the anchor)
- A "negative" (an item different from the anchor)

The goal is to teach the model to:

- Push the anchor and positive close together in the feature space
- Push the anchor and negative far apart in the feature space

Think of it like organizing a bookshelf:

- You have a fantasy novel (anchor)
- You want to place other fantasy novels (positives) close to it
- You want to place non-fantasy books (negatives) far away from it

Triplet Loss is especially useful for:

- Face recognition (same person = positive, different person = negative)
- Product recommendations (similar products = positive, different products = negative)
- Image search (visually similar images = positive, different images = negative)
- Document similarity (documents on same topic = positive, different topics = negative)

The "margin" parameter controls how far apart the negative examples should be from the anchor
compared to the positive examples.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TripletLossFitnessCalculator(,DataSetType)` | Initializes a new instance of the TripletLossFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Triplet Loss fitness score for the given dataset. |
| `PrepareTripletData(Matrix<>,Vector<>)` | Prepares triplet data (anchor, positive, negative) from the input features and labels. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_margin` | The margin value that determines how far negative examples should be from anchor examples compared to positive examples. |

