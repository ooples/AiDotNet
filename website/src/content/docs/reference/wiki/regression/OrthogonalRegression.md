---
title: "OrthogonalRegression<T>"
description: "Implements orthogonal regression (also known as total least squares), which minimizes the perpendicular  distance from data points to the fitted line or hyperplane."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements orthogonal regression (also known as total least squares), which minimizes the perpendicular 
distance from data points to the fitted line or hyperplane.

## How It Works

Unlike ordinary least squares regression which minimizes vertical distances, orthogonal regression 
minimizes the perpendicular (orthogonal) distance from each data point to the regression line or hyperplane.
This approach is more appropriate when both dependent and independent variables contain measurement errors.

The algorithm works by centering the data, optionally scaling the variables, and then finding the 
solution using matrix decomposition methods such as SVD (Singular Value Decomposition).

For Beginners:
Orthogonal regression is useful when you're not sure which variable is dependent and which is independent,
or when both variables have measurement errors. Think of it as finding the line that's as close as possible
to all points when measuring distance perpendicular to the line, rather than just vertically.

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
    .ConfigureModel(new OrthogonalRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained OrthogonalRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OrthogonalRegression(OrthogonalRegressionOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the OrthogonalRegression class with the specified options and regularization. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the Orthogonal Regression model with the same configuration. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `GetOptions` |  |
| `Serialize` | Gets the type of the model. |
| `Train(Matrix<>,Vector<>)` | Trains the orthogonal regression model on the provided data. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the orthogonal regression model. |

