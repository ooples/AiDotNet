---
title: "BetaRegression<T>"
description: "Beta Regression for modeling proportions and rates bounded in (0, 1)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Beta Regression for modeling proportions and rates bounded in (0, 1).

## For Beginners

When you need to predict proportions (like percentages),
regular regression can give impossible results (negative values or values > 1).
Beta Regression fixes this by:

1. Always producing valid predictions between 0 and 1
2. Naturally handling skewed proportions
3. Allowing varying uncertainty (some predictions more reliable than others)

Example applications:

- Predicting market share (e.g., "37% market share")
- Modeling test pass rates
- Estimating probability scores
- Analyzing biological concentrations

The model uses a "link function" to transform proportions to a scale where linear
modeling works, then transforms predictions back to valid proportions.

## How It Works

Beta Regression is the appropriate choice when your response variable is a continuous
proportion or rate that must fall strictly between 0 and 1. It uses the Beta distribution
and can model both the mean and precision as functions of covariates.

Reference: Ferrari, S.L.P., Cribari-Neto, F. (2004). "Beta Regression for
Modelling Rates and Proportions". Journal of Applied Statistics, 31(7), 799-815.

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
    .ConfigureModel(new BetaRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained BetaRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BetaRegression(BetaRegressionOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of BetaRegression. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MeanCoefficients` | Gets the mean model coefficients. |
| `MeanIntercept` | Gets the mean model intercept. |
| `NumberOfTrees` |  |
| `Precision` | Gets the precision (or its intercept if constant). |
| `PrecisionCoefficients` | Gets the precision model coefficients (if variable precision is enabled). |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateFeatureImportancesAsync(Int32)` |  |
| `ComputeLogLikelihood(Vector<>,Vector<>,Vector<>)` | Computes the log-likelihood. |
| `ComputePredictions(Matrix<>)` | Computes mean (μ) and precision (φ) predictions for all samples. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `InitializeParameters(Vector<>)` | Initializes parameters from target values. |
| `InverseLinkFunction(Double)` | Applies the inverse link function. |
| `LinkFunction(Double)` | Applies the link function. |
| `LinkFunctionDerivative(Double)` | Computes the derivative of the link function. |
| `PredictAsync(Matrix<>)` |  |
| `PredictDistributionsAsync(Matrix<>)` | Predicts full Beta distributions for each input sample. |
| `PredictIntervalAsync(Matrix<>,Double)` | Gets prediction intervals for each input sample. |
| `Serialize` |  |
| `TrainAsync(Matrix<>,Vector<>)` |  |
| `UpdateCoefficientsWLS(Matrix<>,Vector<>,Vector<>,Vector<>,)` | Updates coefficients using weighted least squares. |
| `UpdateMeanModel(Matrix<>,Vector<>,Vector<>,Vector<>)` | Updates the mean model using Fisher scoring. |
| `UpdatePrecisionModel(Matrix<>,Vector<>,Vector<>,Vector<>)` | Updates the precision model using Fisher scoring. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_meanCoefficients` | Coefficients for the mean (μ) model. |
| `_meanIntercept` | Intercept for the mean model. |
| `_numFeatures` | Number of features. |
| `_options` | Configuration options. |
| `_precisionCoefficients` | Coefficients for the precision (φ) model (if variable precision). |
| `_precisionIntercept` | Intercept for the precision model. |
| `_yMin` | Y min-max scaling for mapping to (0,1). |

