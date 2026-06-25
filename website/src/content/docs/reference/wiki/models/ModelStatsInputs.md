---
title: "ModelStatsInputs<T, TInput, TOutput>"
description: "Represents a container for inputs needed to calculate various statistics and metrics for a model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Inputs`

Represents a container for inputs needed to calculate various statistics and metrics for a model.

## For Beginners

This class collects all the data needed to evaluate how well a model performs.

Think of it like gathering all the ingredients before baking:

- Actual values (what really happened)
- Predicted values (what the model thought would happen)
- Feature data (the information used to make predictions)
- The model itself and its parameters

This organized collection makes it easier to calculate accuracy metrics, perform 
diagnostic tests, and visualize model performance without having to pass many
separate parameters around.

## How It Works

This class holds the data and parameters necessary for evaluating model performance and calculating
statistics. It includes actual and predicted values, feature information, the model itself, and
optional functions for fitting and prediction.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModelStatsInputs` | Initializes a new instance of the `ModelStatsInputs` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Actual` | Gets or sets the actual observed values. |
| `Coefficients` | Gets or sets the coefficient values for the model. |
| `FeatureCount` | Gets or sets the number of features or predictor variables in the model. |
| `FeatureNames` | Gets or sets the names of the features used in the model. |
| `FeatureValues` | Gets or sets the values for each feature organized by feature name. |
| `FitFunction` | Gets or sets a function that fits a model to data. |
| `Model` | Gets or sets the predictive model used to generate the statistics. |
| `Predicted` | Gets or sets the values predicted by the model. |
| `XMatrix` | Gets or sets the input data used for predictions. |

