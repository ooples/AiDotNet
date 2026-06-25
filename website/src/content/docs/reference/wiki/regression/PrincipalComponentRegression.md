---
title: "PrincipalComponentRegression<T>"
description: "Implements Principal Component Regression (PCR), a technique that combines principal component analysis (PCA)  with linear regression to handle multicollinearity in the predictor variables."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements Principal Component Regression (PCR), a technique that combines principal component analysis (PCA) 
with linear regression to handle multicollinearity in the predictor variables.

## For Beginners

Think of PCR as a two-step process: first, it finds the most important patterns in your input data 
(principal components), then it uses these patterns instead of the original variables to build a regression model. 
This can help when your original variables are highly related to each other (multicollinear), which can cause 
problems in standard regression.

## How It Works

Principal Component Regression works by first performing principal component analysis (PCA) on the predictor 
variables to reduce their dimensionality, then using these principal components as predictors in a linear 
regression model. This approach is particularly useful when dealing with multicollinearity (high correlation 
among predictor variables) or when the number of predictors is large relative to the number of observations.

The algorithm first centers and scales the data, performs PCA to extract principal components, selects a 
subset of these components based on either a fixed number or explained variance ratio, and then performs 
linear regression using the selected components.

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
    .ConfigureModel(new PrincipalComponentRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained PrincipalComponentRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrincipalComponentRegression(PrincipalComponentRegressionOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the PrincipalComponentRegression class with the specified options and regularization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Trains the principal component regression model on the provided data. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateFeatureImportances` | Gets the type of the model. |
| `Clone` | Makes predictions for the given input data. |
| `CreateNewInstance` | Creates a new instance of the Principal Component Regression model with the same configuration. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetOptions` |  |
| `PerformPCA(Matrix<>)` | Performs Principal Component Analysis (PCA) on the input data. |
| `SelectNumberOfComponents(Vector<>)` | Selects the appropriate number of principal components to use in the regression model. |
| `Serialize` | Serializes the model to a byte array. |
| `ValidateInputs(Matrix<>,Vector<>)` | Validates the input data before training. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_components` | The principal components extracted from the training data. |
| `_options` | Configuration options for the principal component regression model. |
| `_xMean` | The mean of each predictor variable used for centering. |
| `_xStd` | The standard deviation of each predictor variable used for scaling. |
| `_yMean` | The mean of the target variable used for centering. |
| `_yStd` | The standard deviation of the target variable used for scaling. |

