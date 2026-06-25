---
title: "MixedEffectsModel"
description: "Mixed-Effects (Hierarchical/Multilevel) Linear Model for clustered and hierarchical data."
section: "Reference"
---

_Regression Models_

Mixed-Effects (Hierarchical/Multilevel) Linear Model for clustered and hierarchical data.

## For Beginners

Use this model when your data has groups or clusters: **Model structure:** y_ij = X_ij * β + Z_ij * u_j + ε_ij Where: - y_ij: Outcome for observation i in group j - X_ij * β: Fixed effects (same for everyone) - Z_ij * u_j: Random effects (vary by group) - ε_ij: Residual error **Example - Student test scores:** score = study_hours * β + (school_intercept + study_hours * school_slope) + error Fixed effects tell you: "On average, each study hour adds β points" Random effects tell you: "School 1 starts 5 points higher, School 2 has steeper study benefit" **Key outputs:** - Fixed effect coefficients and their standard errors - Random effect predictions (BLUPs) for each group - Variance components showing how much groups vary

## How It Works

Mixed-effects models combine fixed effects (population-level patterns) with random effects (group-level variations). They properly account for the correlation structure in hierarchical data, providing correct inference when observations are not independent.

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

