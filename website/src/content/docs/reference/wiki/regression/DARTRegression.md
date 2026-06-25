---
title: "DARTRegression<T>"
description: "DART (Dropouts meet Multiple Additive Regression Trees) regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

DART (Dropouts meet Multiple Additive Regression Trees) regression.

## For Beginners

DART is like gradient boosting with a twist - it randomly "forgets"
some of its trees when learning new ones. This prevents the model from becoming too
specialized and helps it work better on new data.

Key concepts:

- Dropout: Randomly removing trees during training (like dropout in neural networks)
- Normalization: Adjusting predictions after dropout to maintain correct scale
- Ensemble: The final prediction uses all trees (no dropout at prediction time)

When to use DART over regular gradient boosting:

- Your model overfits (training error low, validation error high)
- You want more robust predictions
- You have enough time (DART is slower than regular boosting)

## How It Works

DART applies dropout regularization to gradient boosting. During each iteration, a random
subset of existing trees is dropped, and the new tree is fitted to residuals computed
only from the non-dropped trees. This prevents overfitting and improves generalization.

Reference: Rashmi, K.V. & Gilad-Bachrach, R. (2015). "DART: Dropouts meet Multiple
Additive Regression Trees". AISTATS 2015.

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
    .ConfigureModel(new DARTRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained DARTRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DARTRegression(DARTOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of DART regression. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumberOfTrees` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AccumulateFeatureImportance(DARTRegression<>.DARTTree,Vector<>)` | Recursively accumulates feature importance from a tree. |
| `CalculateFeatureImportancesAsync(Int32)` |  |
| `ComputeNormalizationFactor(Int32[])` | Computes normalization factor for dropout. |
| `ComputePartialPredictions(Matrix<>,Int32[])` | Computes predictions using only specified trees. |
| `CreateLeaf(Vector<>,Int32[])` | Creates a leaf node. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `FindBestSplit(Matrix<>,Vector<>,Int32[],Int32[])` | Finds the best split for a node. |
| `FitTree(Matrix<>,Vector<>,Int32[],Int32[],Int32)` | Fits a decision tree to the data. |
| `GetDropProbability(Int32)` | Gets the drop probability for a tree based on dropout mode. |
| `GetModelMetadata` |  |
| `GetTreePredictions(Vector<>)` | Gets the individual tree predictions for a sample. |
| `GetTreeWeights` | Gets the tree weights. |
| `Predict(Matrix<>)` | Predicts values for input samples. |
| `PredictAsync(Matrix<>)` |  |
| `PredictTree(DARTRegression<>.DARTTree,Matrix<>,Int32)` | Makes a prediction using a single tree for a row of a matrix. |
| `PredictTree(DARTRegression<>.DARTTree,Vector<>)` | Makes a prediction using a single tree for a vector input. |
| `SampleData(Int32)` | Samples data points for this tree. |
| `SampleFeatures` | Samples features for this tree. |
| `SelectDroppedTrees(Int32)` | Selects trees to drop for this iteration. |
| `Serialize` |  |
| `SplitData(Matrix<>,Int32[],Int32,)` | Splits data based on a feature and threshold. |
| `TrainAsync(Matrix<>,Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numFeatures` | Number of features. |
| `_options` | Configuration options. |
| `_random` | Random number generator. |
| `_treeWeights` | Tree weights (may differ after normalization). |
| `_trees` | Individual tree structures. |

