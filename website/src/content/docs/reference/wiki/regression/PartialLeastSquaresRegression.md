---
title: "PartialLeastSquaresRegression"
description: "Implements Partial Least Squares Regression (PLS), a technique that combines features from principal  component analysis and multiple linear regression to handle situations with many correlated predictors."
section: "Reference"
---

_Regression Models_

Implements Partial Least Squares Regression (PLS), a technique that combines features from principal 
component analysis and multiple linear regression to handle situations with many correlated predictors.

## How It Works

Partial Least Squares Regression is particularly useful when dealing with many predictor variables 
that may be highly correlated. It works by finding a linear combination of the predictors (components) 
that maximizes the covariance between the predictors and the response variable.

Unlike Principal Component Regression which only considers the variance in the predictor variables, 
PLS regression considers both the variance in the predictors and their relationship with the response variable.
This often leads to models with better predictive power, especially when the predictors are highly correlated.

For Beginners:
Think of PLS regression as a way to find the most important patterns in your input data that are also 
strongly related to what you're trying to predict. It's like finding the key ingredients in a recipe 
that most influence the taste, rather than just the most abundant ingredients.

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
    .ConfigureModel(new PartialLeastSquaresRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained PartialLeastSquaresRegression.");
```

