---
title: "IModel<TInput, TOutput, TMetadata>"
description: "Defines the core functionality for machine learning models that can be trained on data and make predictions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the core functionality for machine learning models that can be trained on data and make predictions.

## How It Works

This interface represents the fundamental operations that all machine learning models should support:
training on data, making predictions with new data, and providing metadata about the model's performance.

**For Beginners:** A machine learning model is like a recipe that learns from examples.

Think of a model as a student learning to recognize patterns:

- First, you train the model by showing it examples (training data)
- Then, the model learns patterns from these examples
- Finally, when given new information, the model uses what it learned to make predictions

For example, if you want to predict house prices:

- You train the model with data about houses (size, location, etc.) and their prices
- The model learns the relationship between house features and prices
- When given information about a new house, it predicts what the price might be

This interface provides the essential methods needed for this learning and prediction process.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetModelMetadata` | Retrieves metadata and performance metrics about the trained model. |
| `Predict()` | Uses the trained model to make predictions for new input data. |
| `Train(,)` | Trains the model using input features and their corresponding target values. |

