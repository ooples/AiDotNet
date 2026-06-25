---
title: "DecisionTreeRegression"
description: "Represents a decision tree regression model that predicts continuous values based on input features."
section: "Reference"
---

_Regression Models_

Represents a decision tree regression model that predicts continuous values based on input features.

## For Beginners

A decision tree regression is like a flowchart that helps predict numerical values. Think of it like answering a series of yes/no questions to reach a prediction: - "Is the temperature above 75—F?" - "Is the humidity below 50%?" - "Is it a weekend?" Each question splits the data into two groups, and the tree learns which questions to ask to make the most accurate predictions. For example, a decision tree might predict house prices based on features like square footage, number of bedrooms, and neighborhood. The model is called a "tree" because it resembles an upside-down tree, with a single starting point (root) that branches out into multiple endpoints (leaves) where the final predictions are made.

## How It Works

Decision tree regression builds a model in the form of a tree structure where each internal node represents a decision based on a feature, each branch represents an outcome of that decision, and each leaf node represents a predicted value. The model is trained by recursively splitting the data based on the optimal feature and threshold that minimizes the prediction error.

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 1.0, 2.0 }, new[] { 2.0, 3.0 }, new[] { 3.0, 4.0 },
    new[] { 4.0, 5.0 }, new[] { 5.0, 6.0 }, new[] { 6.0, 7.0 }
};
double[] targets = { 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new DecisionTreeRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained DecisionTreeRegression.");
```

