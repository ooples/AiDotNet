---
title: "PrincipalComponentRegression"
description: "Implements Principal Component Regression (PCR), a technique that combines principal component analysis (PCA)  with linear regression to handle multicollinearity in the predictor variables."
section: "Reference"
---

_Regression Models_

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

