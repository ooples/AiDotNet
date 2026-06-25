---
title: "BootstrapResult<T>"
description: "Represents the results of bootstrap validation for a machine learning model, containing R² metrics for training, validation, and test datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents the results of bootstrap validation for a machine learning model, containing R² metrics
for training, validation, and test datasets.

## For Beginners

This class stores how well a model performs on different parts of your data.

When evaluating machine learning models:

- It's important to know how well they perform on different datasets
- Bootstrap validation creates multiple samples from your data to test model stability
- R² (R-squared) is a common metric that measures how well your model explains the variation in the data

This class stores three R² values:

- Training R²: How well the model fits the data it was trained on
- Validation R²: How well the model performs on data used for tuning
- Test R²: How well the model generalizes to completely new data

These values help you understand:

- If your model is underfitting (low R² on all datasets)
- If your model is overfitting (high training R², much lower test R²)
- How stable your model's performance is across different data splits

## How It Works

Bootstrap validation is a resampling technique used to evaluate machine learning models by repeatedly 
sampling from the available data with replacement. This class stores the R² (coefficient of determination) 
values for different data splits in the bootstrap process. R² measures the proportion of variance in the 
dependent variable that is predictable from the independent variables, with values ranging from 0 to 1 
where higher values indicate better model fit. The class uses generic type parameter T to support different 
numeric types for the R² values, such as float, double, or decimal.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BootstrapResult` | Initializes a new instance of the BootstrapResult class with all R² values set to zero. |

## Properties

| Property | Summary |
|:-----|:--------|
| `TestR2` | Gets or sets the R² value for the test dataset. |
| `TrainingR2` | Gets or sets the R² value for the training dataset. |
| `ValidationR2` | Gets or sets the R² value for the validation dataset. |

