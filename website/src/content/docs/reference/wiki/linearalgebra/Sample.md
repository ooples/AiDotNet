---
title: "Sample<T>"
description: "Represents a single data sample consisting of features and a target value for machine learning algorithms."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LinearAlgebra`

Represents a single data sample consisting of features and a target value for machine learning algorithms.

## For Beginners

A Sample is like a single example that we use to train or test a machine learning model.

Think of it this way: if you were teaching someone to identify fruits, each Sample would be one fruit.
The "Features" would be the characteristics you observe (color, size, shape, texture),
and the "Target" would be the correct answer (apple, banana, orange).

For instance, in a house price prediction model:

- Features might include: number of bedrooms, square footage, neighborhood rating, etc.
- Target would be the actual price of the house

The machine learning algorithm learns from many of these samples to make predictions on new data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Sample(Vector<>,)` | Initializes a new instance of the Sample class with the specified features and target. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Features` | Gets or sets the feature vector containing the input values for this sample. |
| `Target` | Gets or sets the target value (or label) for this sample. |

