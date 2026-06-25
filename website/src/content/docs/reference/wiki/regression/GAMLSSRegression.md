---
title: "GAMLSSRegression<T>"
description: "GAMLSS (Generalized Additive Models for Location, Scale, and Shape) regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

GAMLSS (Generalized Additive Models for Location, Scale, and Shape) regression.

## For Beginners

Traditional regression models predict a single value (the mean).
GAMLSS predicts an entire probability distribution by modeling multiple parameters:

- Location (μ): Controls where the distribution is centered (like the mean)
- Scale (σ): Controls how spread out the distribution is (like standard deviation)
- Shape (ν, τ): Controls the shape (skewness, tail behavior)

This is powerful because:

1. You get uncertainty estimates that vary with your inputs
2. You can model phenomena where variance depends on the predictors
3. You get proper prediction intervals instead of assuming constant variance

Example use cases:

- Financial forecasting where volatility depends on market conditions
- Medical studies where patient variability depends on treatment
- Any scenario where "it depends" applies to uncertainty, not just the average

## How It Works

GAMLSS extends generalized linear models by allowing any or all distribution parameters
to be modeled as functions of the explanatory variables. This enables heteroskedastic
modeling and full distributional regression.

Reference: Rigby, R.A., Stasinopoulos, D.M. (2005). "Generalized additive models
for location, scale and shape". Applied Statistics, 54, 507-554.

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
    .ConfigureModel(new GAMLSSRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained GAMLSSRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GAMLSSRegression(GAMLSSOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of GAMLSSRegression. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LocationCoefficients` | Gets the location (mean) model coefficients. |
| `NumberOfTrees` |  |
| `ScaleCoefficients` | Gets the scale model coefficients. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateFeatureImportancesAsync(Int32)` |  |
| `ClampLogScale()` | Clamps a log-link (scale/shape) linear-predictor value to the numerically safe range [`MinLogScale`, `MaxLogScale`]. |
| `Clone` | Creates a deep copy via serialization to preserve private coefficient state. |
| `ComputeDeviance(Vector<>,Vector<>,Vector<>,Vector<>)` | Computes the deviance (negative log-likelihood). |
| `CreateDistribution(,,)` | Creates a distribution with the specified parameters. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `FitLocationParameter(Matrix<>,Vector<>,Vector<>,Vector<>,Vector<>)` | Fits the location parameter using IRLS (Iteratively Reweighted Least Squares). |
| `FitScaleParameter(Matrix<>,Vector<>,Vector<>,Vector<>,Vector<>)` | Fits the scale parameter using IRLS. |
| `FitShapeParameter(Matrix<>,Vector<>,Vector<>,Vector<>,Vector<>)` | Fits the shape parameter using IRLS. |
| `GetModelMetadata` |  |
| `InitializeParameters(Vector<>)` | Initializes distribution parameters from the target values. |
| `PredictAsync(Matrix<>)` |  |
| `PredictDistributionsAsync(Matrix<>)` | Predicts full probability distributions for each input sample. |
| `PredictIntervalAsync(Matrix<>,Double)` | Gets prediction intervals for each input sample. |
| `PredictParameter(Vector<>,Vector<>,,Boolean)` | Predicts a single parameter value for one sample. |
| `Serialize` |  |
| `SolveLinearSystem(Matrix<>,Vector<>,Int32)` | Solves a linear system using Gaussian elimination. |
| `UpdateCoefficients(Matrix<>,Vector<>,Vector<>,Vector<>,)` | Updates coefficients using weighted least squares. |
| `UpdateLinearPredictor(Matrix<>,Vector<>,Vector<>,,Boolean)` | Updates the linear predictor for all samples. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_locationCoefficients` | Coefficients for the location parameter model. |
| `_locationIntercept` | Intercept for the location parameter. |
| `_numFeatures` | Number of features. |
| `_options` | Configuration options. |
| `_scaleCoefficients` | Coefficients for the scale parameter model. |
| `_scaleIntercept` | Intercept for the scale parameter. |
| `_shapeCoefficients` | Coefficients for the shape parameter model (if applicable). |
| `_shapeIntercept` | Intercept for the shape parameter. |
| `_yMean` | Y standardization for scale-invariant training. |

