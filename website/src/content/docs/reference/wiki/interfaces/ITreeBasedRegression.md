---
title: "ITreeBasedRegression<T>"
description: "Defines the core functionality for tree-based machine learning models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the core functionality for tree-based machine learning models.

## How It Works

Tree-based models make predictions by following a series of decision rules organized in a tree-like structure.
These models can be used for both classification (predicting categories) and regression (predicting numeric values).

**For Beginners:** Tree-based models work like a flowchart of yes/no questions to make predictions.
Imagine you're trying to predict if someone will like a movie:

1. Is it an action movie? If yes, go to question 2. If no, go to question 3.
2. Does it have their favorite actor? If yes, predict "Like". If no, predict "Dislike".
3. Is it less than 2 hours long? If yes, predict "Like". If no, predict "Dislike".

This is a simple decision tree. More advanced tree-based models like Random Forests or 
Gradient Boosted Trees use multiple trees together to make better predictions.

This interface inherits from IFullModel<T>, which provides the basic methods for training,
predicting, and evaluating machine learning models.

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureImportances` | Gets the relative importance of each feature in making predictions. |
| `MaxDepth` | Gets the maximum depth (number of sequential decisions) allowed in each decision tree. |
| `NumberOfTrees` | Gets the number of decision trees used in the model. |

