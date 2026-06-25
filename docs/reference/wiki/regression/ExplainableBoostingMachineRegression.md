---
title: "ExplainableBoostingMachineRegression"
description: "Explainable Boosting Machine (EBM) for interpretable regression."
section: "Reference"
---

_Regression Models_

Explainable Boosting Machine (EBM) for interpretable regression.

## For Beginners

EBM is special because it gives you the best of both worlds: - High accuracy (comparable to gradient boosting and random forests) - Full interpretability (you can see exactly why each prediction was made) How it works: 1. For each feature, EBM learns a "shape function" that shows how that feature affects the prediction 2. The final prediction is simply the sum of all these shape functions plus an intercept 3. You can plot these shape functions to understand exactly how the model uses each feature For example, in predicting house prices: - The shape function for "square footage" might show a linear increase - The shape function for "age" might show older houses have lower prices - The prediction for a specific house is just: intercept + f(sqft) + f(age) + ... This additive structure makes EBM uniquely interpretable while still being accurate.

## How It Works

EBM is a Generalized Additive Model (GAM) with boosting that provides glass-box interpretability while maintaining high accuracy. It learns smooth functions for each feature and optionally pairwise interactions. 

Reference: Lou, Y., et al. "Intelligible Models for Healthcare: Predicting Pneumonia Risk and Hospital 30-day Readmission" (2012).

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
    .ConfigureModel(new ExplainableBoostingMachineRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained ExplainableBoostingMachineRegression.");
```

