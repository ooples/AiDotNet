---
title: "RegularizationOptions"
description: "Configuration options for regularization techniques used to prevent overfitting in machine learning models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models`

Configuration options for regularization techniques used to prevent overfitting in machine learning models.

## For Beginners

Regularization helps prevent your model from "memorizing" the training data.

Think of regularization like training wheels for your machine learning model:

- Without regularization, your model might become too complex and "overfit" the training data
- Overfitting means your model performs well on training data but poorly on new data
- Regularization adds constraints that keep your model simpler and more general

There are three main types of regularization you can choose:

- L1 (Lasso): Tends to create sparse models by setting some weights exactly to zero
- L2 (Ridge): Keeps all weights small but non-zero
- Elastic Net: A mix of both L1 and L2 approaches

For example, if you're predicting house prices:

- Without regularization: Your model might put too much importance on rare features like "has a wine cellar"
- With regularization: Your model focuses more on common, reliable patterns like square footage and location

This class lets you configure what type of regularization to use and how strongly to apply it.

## How It Works

Regularization is a technique used in machine learning to prevent overfitting by adding a penalty term to the 
loss function. Overfitting occurs when a model learns the training data too well, including its noise and 
outliers, resulting in poor performance on new, unseen data. By adding regularization, the model is 
encouraged to learn simpler patterns, improving its generalization capabilities. This class provides 
configuration options for different types of regularization methods, including L1 (Lasso), L2 (Ridge), 
and Elastic Net regularization, allowing users to control the strength and behavior of the regularization 
applied to their models.

## Properties

| Property | Summary |
|:-----|:--------|
| `L1Ratio` | Gets or sets the mixing ratio between L1 and L2 regularization when using Elastic Net. |
| `Strength` | Gets or sets the strength of the regularization penalty. |
| `Type` | Gets or sets the type of regularization to apply to the model. |

