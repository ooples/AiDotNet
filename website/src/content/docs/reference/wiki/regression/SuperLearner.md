---
title: "SuperLearner<T>"
description: "Super Learner (Stacking) ensemble for optimal model combination."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Super Learner (Stacking) ensemble for optimal model combination.

## For Beginners

Super Learner is an ensemble technique that:

1. Takes multiple different models (your "library" of algorithms)
2. Uses cross-validation to see how well each model predicts
3. Learns the best way to combine their predictions
4. Creates a final model that's at least as good as the best individual model

**Key advantage:** You don't have to choose which model is best - Super Learner
figures that out automatically and combines them optimally.

**Example usage:**

- Add a linear model (handles linear relationships)
- Add a random forest (handles interactions)
- Add a neural network (handles complex patterns)
- Super Learner learns to use each when appropriate

## How It Works

Super Learner combines multiple base models using cross-validated predictions to train
an optimal meta-learner. It's proven to perform at least as well as the best single
base learner (oracle inequality).

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
    .ConfigureModel(new SuperLearner<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained SuperLearner.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SuperLearner` | Initializes a new instance with a default base model. |
| `SuperLearner(IEnumerable<IFullModel<,Matrix<>,Vector<>>>,SuperLearnerOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of Super Learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | SuperLearner is an ensemble that doesn't support optimizer parameter injection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddBaseModel(IFullModel<,Matrix<>,Vector<>>)` | Adds a base model to the library. |
| `CloneModel(IFullModel<,Matrix<>,Vector<>>)` | Clones a model by creating a new instance. |
| `CreateInstance` |  |
| `Deserialize(Byte[])` |  |
| `ExtractRows(Matrix<>,Int32[])` | Extracts rows from a matrix. |
| `ExtractValues(Vector<>,Int32[])` | Extracts values from a vector. |
| `GenerateFoldIndices(Int32)` | Generates fold indices for cross-validation. |
| `GetActiveFeatureIndices` | Returns all features since the ensemble uses sub-models on all features. |
| `GetCVPerformance` | Gets the cross-validation performance (MSE) of each base model. |
| `GetFoldSplit(Int32[],Int32)` | Gets the train/validation split for a fold. |
| `GetMetaWeights` | Gets the meta-learner weights for each base model. |
| `GetModelContributions` | Gets the contribution of each base model based on weights. |
| `GetModelMetadata` |  |
| `InvertMatrix(Matrix<>)` | Simple matrix inversion using Gaussian elimination. |
| `NormalizeMetaFeatures(Matrix<>)` | Normalizes meta-features (base model predictions). |
| `NormalizeMetaFeaturesForPrediction(Matrix<>)` | Normalizes meta-features using stored means/stds. |
| `OptimizeModel(Matrix<>,Vector<>)` |  |
| `Predict(Matrix<>)` |  |
| `PredictSingle(Vector<>)` |  |
| `Serialize` |  |
| `Train(Matrix<>,Vector<>)` |  |
| `TrainLasso(Matrix<>,Vector<>)` | Lasso regression meta-learner (coordinate descent). |
| `TrainLinearRegression(Matrix<>,Vector<>)` | Linear regression meta-learner. |
| `TrainMetaLearner(Matrix<>,Vector<>)` | Trains the meta-learner. |
| `TrainNNLS(Matrix<>,Vector<>)` | Non-negative least squares. |
| `TrainPerformanceWeighted(Int32)` | Performance-weighted averaging. |
| `TrainRidge(Matrix<>,Vector<>)` | Ridge regression meta-learner. |
| `TrainSimpleAverage(Int32)` | Simple averaging (equal weights). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_baseModels` | Base models in the library. |
| `_cvPerformance` | Cross-validation performance of each base model. |
| `_metaIntercept` | Meta-learner intercept. |
| `_metaWeights` | Meta-learner weights or coefficients. |
| `_numFeatures` | Number of features. |
| `_options` | Configuration options. |
| `_predMeans` | Means of base model predictions (for normalization). |
| `_predStds` | Standard deviations of base model predictions (for normalization). |
| `_random` | Random number generator. |

