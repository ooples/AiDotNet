---
title: "MultipleRegression"
description: "Represents a multiple linear regression model that predicts a target value based on multiple input features."
section: "Reference"
---

_Regression Models_

Represents a multiple linear regression model that predicts a target value based on multiple input features.

## For Beginners

Multiple regression is like a formula that predicts one value based on several inputs.

Think of it like a house price calculator:

- You provide information like square footage, number of bedrooms, neighborhood rating, etc.
- Each feature has a certain importance (called a coefficient)
- The model combines all these factors with their importances to make a prediction

For example, the formula might be:
House Price = $50,000 + ($100 × Square Footage) + ($15,000 × Number of Bedrooms) + ($25,000 × Neighborhood Rating)

The model learns the best values for these coefficients from your training data to make accurate predictions.

## How It Works

Multiple linear regression extends simple linear regression to incorporate multiple input features. It models the
relationship between several independent variables and one dependent variable by fitting a linear equation to the
observed data. The model assumes that the relationship between inputs and the output is linear, meaning that the output
can be calculated as a weighted sum of the input features plus a constant term (intercept).

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
    .ConfigureModel(new MultipleRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained MultipleRegression.");
```

