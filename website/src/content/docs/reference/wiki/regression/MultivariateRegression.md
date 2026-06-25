---
title: "MultivariateRegression<T>"
description: "Represents a multivariate linear regression model that predicts a target value based on multiple input features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Represents a multivariate linear regression model that predicts a target value based on multiple input features.

## For Beginners

Multivariate regression is like a recipe that combines several ingredients to predict an outcome.

Think of it like a car's fuel efficiency calculator:

- You provide information like car weight, engine size, aerodynamics, etc.
- Each factor has a certain importance (coefficient) in determining fuel efficiency
- The model combines all these factors to make a prediction

For example, the formula might be:
Miles per gallon = 35 - (0.005 × Car Weight) - (2 × Engine Size) + (3 × Aerodynamic Rating)

The model learns the best values for these coefficients from your training data to make accurate predictions.

## How It Works

Multivariate linear regression is a statistical method that models the relationship between multiple independent
variables and a dependent variable by fitting a linear equation to the observed data. The model assumes that the
relationship between inputs and the output is linear, meaning that the output can be calculated as a weighted sum
of the input features plus a constant term (intercept) if included.

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
    .ConfigureModel(new MultivariateRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained MultivariateRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultivariateRegression(RegressionOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the `MultivariateRegression` class with optional custom options and regularization. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the Multivariate Regression model with the same configuration. |
| `Predict(Matrix<>)` | Makes predictions for new data points using the trained multivariate regression model. |
| `Train(Matrix<>,Vector<>)` | Trains the multivariate regression model using the provided features and target values. |

