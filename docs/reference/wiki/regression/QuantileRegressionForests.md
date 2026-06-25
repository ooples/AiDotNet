---
title: "QuantileRegressionForests"
description: "Implements Quantile Regression Forests, an extension of Random Forests that can predict conditional quantiles of the target variable, not just the conditional mean."
section: "Reference"
---

_Regression Models_

Implements Quantile Regression Forests, an extension of Random Forests that can predict conditional quantiles of the target variable, not just the conditional mean.

## For Beginners

While standard Random Forests tell you the average prediction, Quantile Regression Forests can tell you about the entire range of possible outcomes. For example, they can predict not just the expected value, but also the 10th percentile (a pessimistic scenario) or the 90th percentile (an optimistic scenario). This is particularly useful when you need to understand the uncertainty in your predictions or when the relationship between variables varies across different parts of the distribution.

## How It Works

Quantile Regression Forests extend the Random Forests algorithm to estimate the full conditional distribution of the response variable, not just its mean. This allows for prediction of any quantile of the response variable, providing a more complete picture of the relationship between predictors and the response. 

The algorithm works by building multiple decision trees on bootstrap samples of the training data, similar to Random Forests. However, instead of averaging the predictions, it uses the empirical distribution of the predictions from all trees to estimate quantiles.

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
    .ConfigureModel(new QuantileRegressionForests<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained QuantileRegressionForests.");
```

