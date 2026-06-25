---
title: "IFitDetector<T, TInput, TOutput>"
description: "Defines an interface for detecting how well a machine learning model fits the data."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines an interface for detecting how well a machine learning model fits the data.

## How It Works

**For Beginners:** This interface helps determine if your model is learning properly or has problems.

When training machine learning models, three common problems can occur:

1. Underfitting: The model is too simple and doesn't capture important patterns in the data.
- Like using only a house's age to predict its price, ignoring size, location, etc.
- Signs: Poor performance on both training and new data

2. Overfitting: The model memorizes the training data instead of learning general patterns.
- Like memorizing specific houses and their prices instead of understanding what makes houses valuable
- Signs: Excellent performance on training data but poor performance on new data

3. Good fit: The model captures the important patterns without memorizing noise.
- Like understanding that location, size, and condition affect house prices
- Signs: Good performance on both training data and new data

This interface provides methods to analyze your model's performance and detect which
of these situations you're dealing with, so you can make appropriate adjustments.

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes model evaluation data to detect whether the model is underfitting, overfitting, or has a good fit. |

