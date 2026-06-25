---
title: "DatasetResult<T, TInput, TOutput>"
description: "Represents detailed results and statistics for a specific dataset (training, validation, or test)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents detailed results and statistics for a specific dataset (training, validation, or test).

## For Beginners

This class stores all the details about how the model performs on a specific dataset.

For each dataset (training, validation, or test), this stores:

- The actual input data (X) and target values (Y)
- The model's predictions
- Various error measurements (how far predictions are from actual values)
- Statistics about prediction quality (how well the model captures patterns)
- Basic statistics about both actual values and predictions

This detailed information helps you:

- Understand exactly how well your model is performing
- Identify specific strengths and weaknesses
- Compare performance across different datasets
- Diagnose issues like overfitting or underfitting

## How It Works

This nested class encapsulates all the data and statistics related to model performance on a specific dataset. 
It includes the input features (X), target values (Y), model predictions, and various statistical measures that 
quantify different aspects of model performance. These statistics include error metrics (such as mean squared error, 
mean absolute error), prediction quality metrics (such as R-squared, correlation), and basic descriptive statistics 
for both the actual and predicted values. This comprehensive collection of information allows for thorough analysis 
of model performance on the dataset.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DatasetResult` | Initializes a new instance of the DatasetResult class with empty data structures. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActualBasicStats` | Gets or sets the basic descriptive statistics for the actual target values. |
| `ErrorStats` | Gets or sets the error statistics for the model's predictions. |
| `PredictedBasicStats` | Gets or sets the basic descriptive statistics for the predicted values. |
| `PredictionStats` | Gets or sets the prediction quality statistics for the model. |
| `Predictions` | Gets or sets the model's predictions for the dataset. |
| `X` | Gets or sets the input feature matrix for the dataset. |
| `Y` | Gets or sets the target values for the dataset. |

