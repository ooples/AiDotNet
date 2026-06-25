---
title: "IPredictiveModel<T, TInput, TOutput>"
description: "Defines the core functionality of a trained predictive model that can make predictions on new data."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the core functionality of a trained predictive model that can make predictions on new data.

## How It Works

This interface represents a machine learning model that has been trained and is ready to use.

**For Beginners:** Think of a predictive model like a calculator that has been specially programmed
to solve a specific type of problem. After you've "trained" it with examples (like showing it
houses and their prices), it can make educated guesses about new examples (predicting prices
for houses it hasn't seen before).

This interface provides the methods you need to:

- Make predictions with your trained model
- Get information about how the model was created and how well it performs

## Methods

| Method | Summary |
|:-----|:--------|
| `GetModelMetadata` | Retrieves metadata and performance information about the trained model. |
| `Predict()` | Makes predictions using the trained model on new input data. |

