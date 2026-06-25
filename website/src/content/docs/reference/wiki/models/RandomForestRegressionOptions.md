---
title: "RandomForestRegressionOptions"
description: "Configuration options for Random Forest Regression, an ensemble learning method that combines multiple decision trees to improve prediction accuracy and control overfitting."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Random Forest Regression, an ensemble learning method that combines
multiple decision trees to improve prediction accuracy and control overfitting.

## For Beginners

Random Forest Regression is like getting predictions from a group of experts instead of just one person.

Think about house price prediction:

- A single decision tree is like asking one real estate agent to estimate a house's value
- A Random Forest is like asking 100 different agents and taking their average estimate
- Each agent (tree) looks at slightly different aspects of the house and has seen different houses before
- The combined wisdom of many agents usually gives a more reliable prediction than any single agent

What this technique does:

- It builds many decision trees (like a "forest")
- Each tree is built using a random sample of your data
- Each tree also considers a random subset of features at each decision point
- The final prediction is the average of all individual tree predictions

This is especially useful when:

- You need more accurate predictions than a single decision tree can provide
- Your data has complex relationships that are hard to capture in one model
- You want to understand which features are most important for prediction
- You're concerned about overfitting (when a model works well on training data but poorly on new data)

For example, in medical diagnosis, a Random Forest might combine the "opinions" of many decision trees
to predict patient outcomes more accurately than any single diagnostic approach.

This class lets you configure how the Random Forest ensemble is constructed.

## How It Works

Random Forest Regression is an ensemble learning technique that constructs multiple decision trees
during training and outputs the average prediction of the individual trees for regression tasks.
This method combines the concepts of bagging (bootstrap aggregating) and feature randomization
to create a diverse set of trees. Each tree is trained on a random bootstrap sample of the original
data, and at each node, only a random subset of features is considered for splitting. These randomization
techniques help reduce correlation between trees, which is essential for the ensemble's performance.
Random Forests generally provide higher accuracy than single decision trees, are more robust to noise
and outliers, handle high-dimensional data well, and are less prone to overfitting. They also provide
built-in estimates of feature importance, making them valuable for feature selection and understanding
the underlying data structure.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumberOfTrees` | Gets or sets the number of trees to grow in the forest. |

