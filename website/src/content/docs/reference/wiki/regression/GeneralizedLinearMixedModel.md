---
title: "GeneralizedLinearMixedModel<T>"
description: "Generalized Linear Mixed-Effects Model (GLMM) for non-Gaussian hierarchical data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression.MixedEffects`

Generalized Linear Mixed-Effects Model (GLMM) for non-Gaussian hierarchical data.

## For Beginners

GLMMs are like mixed models but for outcomes that aren't continuous/normal:

Common use cases:

- Binary outcomes (yes/no): Use logistic GLMM with logit link
- Count data: Use Poisson GLMM with log link
- Overdispersed counts: Use Negative Binomial GLMM
- Proportions: Use Binomial GLMM

Example: Student pass/fail across schools

- Fixed effect: Effect of study time on probability of passing
- Random intercept: Schools have different baseline pass rates
- Random slope: Effect of study time might differ by school

The model estimates effects on the log-odds (or log-rate) scale,
which can then be converted to probabilities or rates.

## How It Works

GLMMs extend linear mixed models to handle non-Gaussian responses (binary, count, etc.)
by incorporating a link function and response distribution from the exponential family.

The model has the form: g(E[y|u]) = X*beta + Z*u
where:

- g() is the link function (logit, log, identity, etc.)
- X*beta: Fixed effects on the linear predictor scale
- Z*u: Random effects on the linear predictor scale
- u ~ N(0, D) where D is the variance-covariance matrix

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression.MixedEffects;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 1.0, 2.0 }, new[] { 2.0, 3.0 }, new[] { 3.0, 4.0 },
    new[] { 4.0, 5.0 }, new[] { 5.0, 6.0 }, new[] { 6.0, 7.0 }
};
double[] targets = { 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new GeneralizedLinearMixedModel<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained GeneralizedLinearMixedModel.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GeneralizedLinearMixedModel(GLMMOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new Generalized Linear Mixed-Effects Model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AIC` | Gets the AIC (Akaike Information Criterion). |
| `BIC` | Gets the BIC (Bayesian Information Criterion). |
| `Dispersion` | Gets the dispersion parameter. |
| `Family` | Gets the response distribution family. |
| `FixedEffects` | Gets the fixed effects coefficients. |
| `LinkFunction` | Gets the link function. |
| `LogLikelihood` | Gets the log-likelihood of the fitted model. |
| `ParameterCount` | Trains the GLMM using Penalized Quasi-Likelihood (PQL) or Laplace approximation. |
| `RandomEffects` | Gets the random effects specifications with estimated values. |
| `VarianceComponents` | Gets the variance decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddRandomIntercept(String,Int32)` | Adds a random intercept effect to the model. |
| `AddRandomSlope(String,Int32,Int32[],Boolean)` | Adds a random slope effect to the model. |
| `ApplyInverseLink(Vector<>)` | Applies the inverse link function to transform from linear predictor to mean. |
| `ApplyLink()` | Applies the link function to transform from mean to linear predictor. |
| `ComputeBLUPs(Matrix<>,Matrix<>,Vector<>,Vector<>)` | Computes BLUPs for random effects. |
| `ComputeLogLikelihood(Vector<>,Vector<>)` | Computes the log-likelihood. |
| `ComputeVarianceDecomposition` | Computes the variance decomposition. |
| `CreateNewInstance` | Gets the model type. |
| `Erf(Double)` | Error function approximation. |
| `ExtractFixedEffectsMatrix(Matrix<>)` | Extracts the fixed effects design matrix. |
| `FitLaplace(Matrix<>,Matrix<>,Vector<>)` | Fits the model using Laplace approximation. |
| `FitPQL(Matrix<>,Matrix<>,Vector<>)` | Fits the model using Penalized Quasi-Likelihood. |
| `FitWeightedLME(Matrix<>,Matrix<>,Vector<>,Vector<>)` | Fits a weighted linear mixed effects model (inner loop of PQL). |
| `GetGroupingColumnCount` | Gets the number of grouping columns. |
| `GetNumberOfParameters` | Gets the number of parameters in the model. |
| `GroupObservations(Matrix<>,Int32)` | Groups observations by grouping variable. |
| `InitializeParameters(Matrix<>,Vector<>)` | Initializes model parameters. |
| `LinkDerivative(Double)` | Computes the derivative of the link function. |
| `LogGamma(Double)` | Log-gamma function approximation. |
| `NormalCDF(Double)` | Standard normal CDF approximation. |
| `NormalPDF(Double)` | Standard normal PDF. |
| `NormalQuantile(Double)` | Standard normal quantile (inverse CDF) approximation. |
| `Predict(Matrix<>)` | Makes predictions for new data on the response scale. |
| `PredictLinearPredictor(Matrix<>)` | Gets predictions on the linear predictor scale (before applying inverse link). |
| `PredictProbability(Matrix<>)` | Gets predicted probabilities for binary classification (logistic GLMM). |
| `SolveOLS(Matrix<>,Vector<>)` | Solves ordinary least squares. |
| `UpdateFixedEffects(Matrix<>,Matrix<>,Vector<>,Vector<>)` | Updates fixed effects coefficients. |
| `UpdateVarianceComponents(Matrix<>,Matrix<>,Vector<>,Vector<>)` | Updates variance components. |
| `VarianceFunction(Double)` | Computes the variance function for the specified family. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |
| `_dispersion` | Dispersion parameter (for overdispersed models). |
| `_fixedEffects` | Fixed effects coefficients. |
| `_logLikelihood` | Log-likelihood of the fitted model. |
| `_nFixedParams` | Number of fixed effect parameters. |
| `_nObservations` | Number of observations. |
| `_options` | Configuration options. |
| `_randomEffects` | List of random effect specifications. |
| `_varianceDecomposition` | Variance decomposition results. |

