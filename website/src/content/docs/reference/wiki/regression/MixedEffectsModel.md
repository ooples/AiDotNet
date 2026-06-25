---
title: "MixedEffectsModel<T>"
description: "Mixed-Effects (Hierarchical/Multilevel) Linear Model for clustered and hierarchical data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Mixed-Effects (Hierarchical/Multilevel) Linear Model for clustered and hierarchical data.

## For Beginners

Use this model when your data has groups or clusters:

**Model structure:**
y_ij = X_ij * β + Z_ij * u_j + ε_ij

Where:

- y_ij: Outcome for observation i in group j
- X_ij * β: Fixed effects (same for everyone)
- Z_ij * u_j: Random effects (vary by group)
- ε_ij: Residual error

**Example - Student test scores:**
score = study_hours * β + (school_intercept + study_hours * school_slope) + error

Fixed effects tell you: "On average, each study hour adds β points"
Random effects tell you: "School 1 starts 5 points higher, School 2 has steeper study benefit"

**Key outputs:**

- Fixed effect coefficients and their standard errors
- Random effect predictions (BLUPs) for each group
- Variance components showing how much groups vary

## How It Works

Mixed-effects models combine fixed effects (population-level patterns) with random effects
(group-level variations). They properly account for the correlation structure in hierarchical
data, providing correct inference when observations are not independent.

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
    .ConfigureModel(new MixedEffectsModel<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained MixedEffectsModel.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MixedEffectsModel(MixedEffectsModelOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of Mixed-Effects Model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | MixedEffects doesn't support optimizer parameter injection. |
| `ResidualVariance` | Gets the residual variance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildRandomEffectDesignMatrix(Matrix<>,Int32[])` | Builds the random effect design matrix for a group. |
| `CenterFeatures(Matrix<>)` | Centers features and stores means. |
| `CenterFeaturesForPrediction(Matrix<>)` | Centers features using stored means. |
| `ComputeBLUPs(Matrix<>,Vector<>,Int32[],Int32[],Vector<>,,Matrix<>)` | Computes Best Linear Unbiased Predictors (BLUPs) for random effects. |
| `ComputeFixedPrediction(Matrix<>,Int32,Vector<>)` | Computes the fixed effects prediction for a row of the data matrix. |
| `ComputeGroupBLUP(Matrix<>,Vector<>,Matrix<>,)` | Computes BLUP for a single group. |
| `ComputeICC` | Computes the intraclass correlation coefficient (ICC). |
| `ComputeStandardErrors(Matrix<>,Vector<>,Int32[],Int32[],Vector<>,,Matrix<>)` | Computes standard errors of fixed effects. |
| `CreateInstance` |  |
| `Deserialize(Byte[])` |  |
| `GetActiveFeatureIndices` | Returns all features used by the model. |
| `GetFixedEffectStandardErrors` | Gets the standard errors of fixed effects. |
| `GetFixedEffects` | Gets the fixed effect coefficients. |
| `GetGroupObservations(Int32[],Int32)` | Gets observation indices for a group. |
| `GetLogLikelihood(Matrix<>,Vector<>,Int32[])` | Gets the log-likelihood of the model. |
| `GetModelMetadata` |  |
| `GetRandomEffectContribution(Matrix<>,Int32,Vector<>)` | Gets the random effect contribution for an observation. |
| `GetRandomEffectVariance` | Gets the random effect variance-covariance matrix. |
| `GetRandomEffects` | Gets the random effects (BLUPs) for each group. |
| `InitializeRandomEffectVariance(Int32)` | Initializes the random effect variance matrix. |
| `InvertMatrix(Matrix<>)` | Simple matrix inversion using Gaussian elimination. |
| `OptimizeModel(Matrix<>,Vector<>)` |  |
| `Predict(Matrix<>)` |  |
| `Predict(Matrix<>,Int32[])` | Predicts values for input samples with known group memberships. |
| `PredictSingle(Vector<>)` |  |
| `Serialize` |  |
| `Train(Matrix<>,Vector<>)` |  |
| `Train(Matrix<>,Vector<>,Int32[])` | Trains the mixed-effects model with explicit group indicators. |
| `UpdateFixedEffects(Matrix<>,Vector<>,Int32[],Dictionary<Int32,Vector<>>,,Matrix<>)` | Updates fixed effects using weighted least squares. |
| `UpdateVarianceComponents(Matrix<>,Vector<>,Int32[],Int32[],Vector<>,Dictionary<Int32,Vector<>>)` | Updates variance components. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_featureMeans` | Feature means for centering. |
| `_fixedEffectStdErrors` | Standard errors of fixed effects. |
| `_fixedEffects` | Fixed effect coefficients. |
| `_groupIndices` | Group indices for training data. |
| `_numFeatures` | Number of features. |
| `_numRandomEffects` | Number of random effect dimensions. |
| `_options` | Configuration options. |
| `_random` | Random number generator. |
| `_randomEffectVariance` | Variance of random effects. |
| `_randomEffects` | Random effect predictions (BLUPs) for each group. |
| `_residualVariance` | Residual variance. |

