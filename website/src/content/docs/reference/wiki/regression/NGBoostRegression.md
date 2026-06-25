---
title: "NGBoostRegression<T>"
description: "NGBoost (Natural Gradient Boosting) for probabilistic regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

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

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NGBoostRegression(NGBoostRegressionOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of NGBoostRegression. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DistributionType` | Gets the distribution type used by this model. |
| `NumberOfTrees` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateFeatureImportancesAsync(Int32)` |  |
| `ComputeMeanScore(Vector<>[],Vector<>)` | Computes the mean score for the current parameter values. |
| `ComputeNaturalGradients(Vector<>[],Matrix<>,Int32)` | Computes natural gradients by preconditioning with Fisher Information. |
| `CreateDistribution(Vector<>)` | Creates a distribution initialized from the target values. |
| `CreateDistributionFromParams(Vector<>[],Int32)` | Creates a distribution from the current parameter values for a specific sample. |
| `CreateDistributionWithParams(Vector<>)` | Creates a distribution with the specified parameters. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `EnsurePositive()` | Ensures a parameter value is positive. |
| `GaussianElimination(Matrix<>,Int32)` | Matrix inversion using Gaussian elimination. |
| `GetModelMetadata` |  |
| `GetSampleIndices(Int32)` | Gets sample indices for subsampling. |
| `InvertMatrix(Matrix<>)` | Inverts a small matrix with regularization. |
| `PredictAsync(Matrix<>)` | Predicts the mean of the distribution for each input sample. |
| `PredictDistributionsAsync(Matrix<>)` | Predicts full probability distributions for each input sample. |
| `PredictIntervalAsync(Matrix<>,Double)` | Gets prediction intervals for each input sample. |
| `PredictQuantilesAsync(Matrix<>,Vector<Double>)` | Predicts quantiles for each input sample. |
| `Serialize` |  |
| `TrainAsync(Matrix<>,Vector<>)` | Trains the NGBoost model using natural gradient boosting. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_initialParameters` | Initial parameter values (e.g., mean of y for location, initial scale). |
| `_numParams` | Number of parameters in the distribution. |
| `_options` | Configuration options. |
| `_scoringRule` | The scoring rule used for optimization. |
| `_trees` | Base learners for each distribution parameter. |
| `_yMean` | Y standardization parameters for scale-invariant training. |

