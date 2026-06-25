---
title: "ZeroInflatedRegression<T>"
description: "Zero-Inflated regression for count data with excess zeros."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Zero-Inflated regression for count data with excess zeros.

## For Beginners

Imagine counting how many times customers visit a store each month:

- Some people NEVER visit (structural zeros) - they live far away or shop elsewhere
- Some people visit sometimes but happened to visit 0 times this month (sampling zeros)

Standard Poisson regression treats all zeros the same, but Zero-Inflated models
recognize these two types of zeros:

1. The "zero model" predicts WHO are structural zeros (π)
2. The "count model" predicts HOW MANY for non-structural-zero people (λ)

Example interpretation:

- "30% of potential customers are 'never visitors' (π = 0.3)"
- "Among potential visitors, the average visit rate is 2.5 times/month (λ = 2.5)"

This gives better predictions and allows you to understand both processes.

## How It Works

Zero-Inflated models handle count data where there are more zeros than a standard
count distribution would predict. They model the data as a mixture: with probability π,
the observation is a "structural zero," and with probability (1-π), it follows a count
distribution (Poisson or Negative Binomial).

Reference: Lambert, D. (1992). "Zero-Inflated Poisson Regression, with an Application
to Defects in Manufacturing". Technometrics, 34(1), 1-14.

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
    .ConfigureModel(new ZeroInflatedRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained ZeroInflatedRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ZeroInflatedRegression(ZeroInflatedRegressionOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of ZeroInflatedRegression. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CountCoefficients` | Gets the count model coefficients. |
| `NumberOfTrees` |  |
| `ZeroCoefficients` | Gets the zero-inflation model coefficients. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateFeatureImportancesAsync(Int32)` |  |
| `ComputeCountProbability(Int32,Double)` | Computes probability from the count distribution. |
| `ComputeLogLikelihood(Vector<>,Vector<>,Vector<>)` | Computes the log-likelihood. |
| `ComputePosteriorZero(Vector<>,Vector<>,Vector<>)` | Computes posterior probability of being a structural zero. |
| `ComputePredictions(Matrix<>)` | Computes predictions for all samples. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `InitializeParameters(Vector<>)` | Initializes parameters from target values. |
| `PredictAsync(Matrix<>)` |  |
| `PredictConditionalCountAsync(Matrix<>)` | Predicts the expected count conditional on not being a structural zero. |
| `PredictPMFAsync(Matrix<>,Int32)` | Predicts the probability mass function for each sample. |
| `PredictZeroProbabilityAsync(Matrix<>)` | Predicts the probability of being a structural zero for each sample. |
| `Serialize` |  |
| `TrainAsync(Matrix<>,Vector<>)` |  |
| `UpdateCoefficientsWLS(Matrix<>,Vector<>,Vector<>,Vector<>,)` | Updates coefficients using weighted least squares. |
| `UpdateCountModel(Matrix<>,Vector<>,Vector<>)` | Updates the count model parameters. |
| `UpdateDispersion(Vector<>,Vector<>,Vector<>)` | Updates dispersion parameter for Negative Binomial. |
| `UpdateZeroModel(Matrix<>,Vector<>)` | Updates the zero-inflation model parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_countCoefficients` | Coefficients for the count model (λ). |
| `_countIntercept` | Intercept for the count model. |
| `_dispersion` | Dispersion parameter for Negative Binomial (if applicable). |
| `_numFeatures` | Number of features. |
| `_options` | Configuration options. |
| `_zeroCoefficients` | Coefficients for the zero-inflation model (π). |
| `_zeroIntercept` | Intercept for the zero-inflation model. |

