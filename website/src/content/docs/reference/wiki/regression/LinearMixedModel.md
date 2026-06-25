---
title: "LinearMixedModel<T>"
description: "Linear Mixed-Effects Model for hierarchical and clustered data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression.MixedEffects`

Linear Mixed-Effects Model for hierarchical and clustered data.

## For Beginners

Mixed models are essential when your data has natural grouping:

Example: Student test scores across 50 schools

- Fixed effect: Effect of study time on scores (same for all schools)
- Random intercept: Each school may have a different baseline score level
- Random slope: Effect of study time might differ by school

Benefits over simple regression:

1. Correct standard errors (not underestimated)
2. Borrowing strength across groups (shrinkage)
3. Quantify between-group variation
4. Handle unbalanced data naturally

When to use mixed models:

- Repeated measures on individuals
- Students in classrooms in schools
- Patients in hospitals
- Longitudinal/panel data

## How It Works

Linear Mixed-Effects (LME) models extend linear regression to handle grouped or
nested data by including both fixed effects (population-level parameters) and
random effects (group-level deviations from the population).

The model has the form: y = X*beta + Z*u + epsilon
where:

- X*beta: Fixed effects (same for all observations)
- Z*u: Random effects (vary by group)
- epsilon: Residual error

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
    .ConfigureModel(new LinearMixedModel<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained LinearMixedModel.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinearMixedModel(LinearMixedModelOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new Linear Mixed-Effects Model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AIC` | Gets the AIC (Akaike Information Criterion). |
| `BIC` | Gets the BIC (Bayesian Information Criterion). |
| `ConditionalRSquared` | Conditional R-squared (fixed + random effects). |
| `FixedEffects` | Gets the fixed effects coefficients. |
| `LogLikelihood` | Gets the log-likelihood of the fitted model. |
| `MarginalRSquared` | Marginal R-squared (fixed effects only). |
| `ParameterCount` | Trains the Linear Mixed-Effects Model using the EM algorithm. |
| `RandomEffects` | Gets the random effects specifications with estimated values. |
| `VarianceComponents` | Gets the variance decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddRandomIntercept(String,Int32)` | Adds a random intercept effect to the model. |
| `AddRandomSlope(String,Int32,Int32[],Boolean)` | Adds a random slope effect to the model. |
| `ComputeBLUPs(Matrix<>,Matrix<>,Vector<>)` | Computes BLUPs (Best Linear Unbiased Predictors) for random effects. |
| `ComputeLogLikelihood(Matrix<>,Matrix<>,Vector<>)` | Computes the log-likelihood. |
| `ComputeRSquaredValues(Matrix<>,Matrix<>,Vector<>)` | Computes marginal and conditional R-squared. |
| `ComputeResiduals(Matrix<>,Vector<>,Vector<>)` | Computes residuals. |
| `ComputeVariance(Vector<>)` | Computes variance of a vector. |
| `ComputeVarianceDecomposition` | Computes the variance decomposition. |
| `CreateNewInstance` | Gets the model type. |
| `ExtractFixedEffectsMatrix(Matrix<>)` | Extracts the fixed effects design matrix. |
| `FitEM(Matrix<>,Matrix<>,Vector<>)` | Fits the model using the EM algorithm. |
| `GetGroupingColumnCount` | Gets the number of grouping columns. |
| `GetNumberOfParameters` | Gets the number of parameters in the model. |
| `GroupObservations(Matrix<>,Int32)` | Groups observations by grouping variable. |
| `InitializeParameters(Matrix<>,Vector<>)` | Initializes model parameters. |
| `Predict(Matrix<>)` | Makes predictions for new data. |
| `SolveOLS(Matrix<>,Vector<>)` | Solves ordinary least squares. |
| `UpdateFixedEffects(Matrix<>,Matrix<>,Vector<>)` | Updates fixed effects coefficients. |
| `UpdateVarianceComponents(Matrix<>,Matrix<>,Vector<>)` | Updates variance components. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |
| `_fixedEffects` | Fixed effects coefficients. |
| `_logLikelihood` | Log-likelihood of the fitted model. |
| `_nFixedParams` | Number of fixed effect parameters. |
| `_nObservations` | Number of observations. |
| `_options` | Configuration options. |
| `_randomEffects` | List of random effect specifications. |
| `_residualVariance` | Residual variance estimate. |
| `_varianceDecomposition` | Variance decomposition results. |

