---
title: "NGBoostRegression"
description: "NGBoost (Natural Gradient Boosting) for probabilistic regression."
section: "Reference"
---

_Regression Models_

NGBoost (Natural Gradient Boosting) for probabilistic regression.

## For Beginners

Traditional regression gives you a point prediction like
"the house price is $300,000." But NGBoost tells you "the house price follows a
normal distribution with mean $300,000 and standard deviation $50,000."

This uncertainty information is valuable because it tells you how confident the
model is. A prediction with small uncertainty means the model is confident.
A prediction with large uncertainty means you should be more cautious.

Key benefits:

- Quantifies prediction uncertainty
- Can use different distributions for different types of data
- Uses natural gradients for stable, efficient learning

## How It Works

NGBoost is an algorithm for probabilistic prediction that uses natural gradients
to efficiently and directly optimize a proper scoring rule. Instead of predicting
a single value, NGBoost predicts a full probability distribution.

Reference: Duan, T., et al. "NGBoost: Natural Gradient Boosting for Probabilistic
Prediction" (2019). https://arxiv.org/abs/1910.03225

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
    .ConfigureModel(new NGBoostRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained NGBoostRegression.");
```

