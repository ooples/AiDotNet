---
title: "PartialLeastSquaresRegression<T>"
description: "Implements Partial Least Squares Regression (PLS), a technique that combines features from principal  component analysis and multiple linear regression to handle situations with many correlated predictors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

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

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PartialLeastSquaresRegression(PartialLeastSquaresRegressionOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the PartialLeastSquaresRegression class with the specified options and regularization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Trains the partial least squares regression model on the provided data. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateFeatureImportances` | Gets the type of the model. |
| `CreateNewInstance` | Creates a new instance of the Partial Least Squares Regression model with the same configuration. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetOptions` |  |
| `Predict(Matrix<>)` | Makes predictions for the given input data. |
| `Serialize` | Serializes the model to a byte array. |
| `ValidateInputs(Matrix<>,Vector<>)` | Validates the input data before training. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_loadings` | The loadings matrix (P) that represents how the original variables load onto the components. |
| `_options` | Configuration options for the partial least squares regression model. |
| `_scores` | The scores matrix (T) that represents the projection of the original data onto the components. |
| `_weights` | The weights matrix (W) used to transform the original variables into components. |
| `_xMean` | The means of the predictor variables used for centering. |
| `_xStd` | The standard deviations of the predictor variables used for scaling. |
| `_yLoadings` | Y-loadings (c) from the NIPALS algorithm: c_k = t_k'*y / (t_k'*t_k). |
| `_yMean` | The mean of the target variable used for centering. |
| `_yStd` | The standard deviation of the target variable used for scaling. |

