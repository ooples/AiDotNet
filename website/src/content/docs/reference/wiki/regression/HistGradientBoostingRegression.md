---
title: "HistGradientBoostingRegression<T>"
description: "Histogram-based Gradient Boosting Regression for fast training on large datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Histogram-based Gradient Boosting Regression for fast training on large datasets.

## For Beginners

Traditional gradient boosting looks at every possible split point
for every feature, which is slow for large datasets. Histogram-based methods group similar
values into "bins" first, then only consider splits between bins.

Think of it like sorting students by height:

- Traditional method: Consider every student's exact height as a potential grouping point
- Histogram method: First group students into height ranges (5'0"-5'2", 5'2"-5'4", etc.),

then only consider splitting between groups

This is much faster because there are far fewer groups than individual heights.

Key advantages:

- 10-100x faster than traditional gradient boosting on large datasets
- Memory efficient (stores bin indices, not raw values)
- Handles missing values naturally
- Similar accuracy to traditional methods

This is the same approach used by LightGBM, XGBoost (hist mode), and scikit-learn's
HistGradientBoostingRegressor.

Usage:

## How It Works

Histogram-based Gradient Boosting discretizes continuous features into a fixed number of bins,
then builds histograms of gradients and hessians for each bin. This approach dramatically
reduces the time complexity of finding the best split from O(n*features) to O(bins*features),
making it suitable for large datasets with millions of samples.

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
    .ConfigureModel(new HistGradientBoostingRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained HistGradientBoostingRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HistGradientBoostingRegression(HistGradientBoostingOptions)` | Initializes a new instance of the HistGradientBoostingRegression class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` | Gets the default loss function used for gradient computation. |
| `FeatureNames` | Gets the model type identifier. |
| `ParameterCount` | Gets the number of parameters in the model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients to update the model. |
| `ApplySplit(HistGradientBoostingRegression<>.HistTreeNode,HistGradientBoostingRegression<>.SplitInfo,[])` | Applies a split to a node, creating left and right children. |
| `BinFeatures(Matrix<>)` | Bins all features in the training data. |
| `BinRow(Matrix<>,Int32)` | Bins a single row of new data for prediction. |
| `BuildHistogram(List<Int32>,Int32,[])` | Builds a gradient histogram for a feature. |
| `BuildTree([],Int32[])` | Builds a single histogram-based tree. |
| `CollectActiveFeatures(HistGradientBoostingRegression<>.HistTreeNode,HashSet<Int32>)` | Recursively collects active feature indices from a tree. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` | Computes gradients without updating parameters. |
| `ComputeQuantileThresholds(List<Double>)` | Computes quantile-based bin thresholds. |
| `CountLeaves(HistGradientBoostingRegression<>.HistTreeNode)` | Counts the number of leaves in a tree. |
| `DeepCopy` | Creates a deep copy of the model. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `DeserializeTree(BinaryReader)` | Deserializes a tree node recursively. |
| `ExportSoftTree(ComputationNode<>,HistGradientBoostingRegression<>.HistTreeNode,Double)` | Exports a soft decision tree as a computation graph. |
| `ExportSoftTreeNode(ComputationNode<>,HistGradientBoostingRegression<>.HistTreeNode,Double)` | Recursively exports a tree node as a soft computation graph. |
| `FindBestSplit(HistGradientBoostingRegression<>.HistTreeNode,[])` | Finds the best split for a node using histograms. |
| `FindBestSplitInHistogram(HistGradientBoostingRegression<>.HistogramBin[],Int32,Int32)` | Finds the best split point within a histogram. |
| `FindBin(Double,List<Double>)` | Finds the bin index for a given value. |
| `GetActiveFeatureIndices` | Gets the indices of features that are actively used by the model. |
| `GetFeatureImportance` | Gets the feature importance scores. |
| `GetFeaturesToConsider` | Gets feature indices to consider for splitting (column subsampling). |
| `GetModelMetadata` | Gets model metadata. |
| `GetOptions` |  |
| `GetParameters` | Gets model parameters. |
| `GetSubsampleIndices(Int32)` | Gets subsample indices for stochastic gradient boosting. |
| `IsFeatureUsed(Int32)` | Checks if a specific feature is used by the model. |
| `LoadModel(String)` | Loads the model from a file. |
| `LoadState(Stream)` | Loads the model state from a stream. |
| `NormalizeFeatureImportances` | Normalizes feature importances to sum to 1. |
| `Predict(Matrix<>)` | Makes predictions for new data. |
| `PredictSingleTree(HistGradientBoostingRegression<>.HistTreeNode,Int32)` | Predicts using a single tree on binned training data. |
| `PredictSingleTreeFromBins(HistGradientBoostingRegression<>.HistTreeNode,Byte[])` | Predicts using a single tree from binned features. |
| `SaveModel(String)` | Saves the model to a file. |
| `SaveState(Stream)` | Saves the model state to a stream. |
| `Serialize` | Serializes the model to a byte array. |
| `SerializeTree(BinaryWriter,HistGradientBoostingRegression<>.HistTreeNode)` | Serializes a tree node recursively. |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` | Sets the active feature indices for the model. |
| `SetLeafValue(HistGradientBoostingRegression<>.HistTreeNode,[])` | Sets the prediction value for a leaf node. |
| `SetParameters(Vector<>)` | Sets model parameters. |
| `Train(Matrix<>,Vector<>)` | Trains the model on the provided data. |
| `WithParameters(Vector<>)` | Creates a new instance with the given parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_activeFeatureIndices` | Active feature indices that are actually used by the model. |
| `_binThresholds` | Bin thresholds for each feature (jagged array). |
| `_binnedData` | Binned feature values for training data. |
| `_defaultLossFunction` | The default loss function for gradient computation. |
| `_featureImportances` | Feature importance scores accumulated during training. |
| `_initialPrediction` | The initial prediction (mean of target values). |
| `_numFeatures` | Number of features in the training data. |
| `_options` | Configuration options for the histogram gradient boosting algorithm. |
| `_random` | Random number generator for subsampling. |
| `_trees` | The collection of histogram-based trees. |

