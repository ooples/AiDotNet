---
title: "PolynomialRegression<T>"
description: "Implements polynomial regression, which extends linear regression by fitting a polynomial equation to the data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements polynomial regression, which extends linear regression by fitting a polynomial equation to the data.

## How It Works

Polynomial regression is useful when the relationship between variables is not linear.
It works by creating new features that are powers of the original features (x, x², x³, etc.),
then applying linear regression techniques to these expanded features.

**For Beginners:** While linear regression fits a straight line to your data,
polynomial regression can fit curves, allowing it to capture more complex patterns.

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
    .ConfigureModel(new PolynomialRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained PolynomialRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PolynomialRegression(PolynomialRegressionOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Creates a new instance of the polynomial regression model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the Polynomial Regression model with the same configuration. |
| `CreatePolynomialFeatures(Matrix<>)` | Creates polynomial features from the original input features. |
| `Predict(Matrix<>)` | Makes predictions using the trained polynomial regression model. |
| `Train(Matrix<>,Vector<>)` | Trains the polynomial regression model using the provided input features and target values. |

