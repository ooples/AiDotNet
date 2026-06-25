---
title: "M5ModelTree<T>"
description: "Represents an M5 model tree for regression problems, combining decision tree structure with linear models at the leaves."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Represents an M5 model tree for regression problems, combining decision tree structure with linear models at the leaves.

## For Beginners

An M5 model tree is like a smart decision-making system for predicting numbers.

Think of it like a flowchart for home price prediction:

- The tree asks questions about the home (Is it bigger than 2000 sq ft? Is it in neighborhood A?)
- Based on the answers, you follow different paths down the tree
- When you reach the end (a leaf), instead of getting a single price value, you get a mini-calculator (linear model)
- This mini-calculator uses the home's features to make a more precise prediction for that specific group of homes

For example, for small homes in urban areas, the price might depend more on location,
while for large homes in suburbs, the number of bathrooms might be more important.
The M5 model tree captures these different patterns for different groups of data.

## How It Works

The M5 model tree is an advanced regression technique that combines the benefits of decision trees and linear regression.
Instead of using a single value at each leaf node (as in standard regression trees), M5 model trees fit linear regression
models at each leaf. This allows the tree to capture both global patterns through its structure and local patterns through
the linear models, often resulting in more accurate predictions compared to standard regression trees.

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
    .ConfigureModel(new M5ModelTree<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained M5ModelTree.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `M5ModelTree(M5ModelTreeOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the `M5ModelTree` class with optional custom options and regularization. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildTreeAsync(Matrix<>,Vector<>,Int32)` | Asynchronously builds the decision tree structure recursively. |
| `CalculateAveragePrediction(DecisionTreeNode<>)` | Calculates the average prediction value for a node. |
| `CalculateFeatureImportancesAsync(Int32)` | Asynchronously calculates the importance of each feature in the model. |
| `CalculateFeatureImportancesRecursiveAsync(DecisionTreeNode<>,)` | Recursively calculates feature importances throughout the tree. |
| `CalculateLeafError(DecisionTreeNode<>)` | Calculates the error if a node were converted to a leaf. |
| `CalculateSubtreeError(DecisionTreeNode<>)` | Calculates the prediction error for a subtree. |
| `CalculateTreeDepth(DecisionTreeNode<>)` | Calculates the maximum depth of the tree. |
| `CollectSamplesFromSubtree(DecisionTreeNode<>,List<Sample<>>)` | Collects all samples from leaf nodes in a subtree. |
| `CountNodes(DecisionTreeNode<>)` | Counts the total number of nodes in the tree. |
| `CreateLeafNode(Matrix<>,Vector<>)` | Creates a leaf node for the decision tree. |
| `CreateNewInstance` | Creates a new instance of the M5ModelTree with the same configuration as the current instance. |
| `Deserialize(Byte[])` | Deserializes the M5 model tree from a byte array, including linear models at leaf nodes. |
| `DeserializeM5Node(BinaryReader)` | Deserializes an M5 tree node including its linear model if present. |
| `FindBestSplitAsync(Matrix<>,Vector<>)` | Asynchronously finds the best feature and threshold to split the data. |
| `FindBestSplitForFeature(Matrix<>,Vector<>,Int32)` | Finds the best threshold for a specific feature to split the data. |
| `FitLinearModel(Matrix<>,Vector<>)` | Fits a linear regression model to the data for a leaf node. |
| `GetModelMetadata` | Gets metadata about the trained model. |
| `GetOptions` |  |
| `PredictAsync(Matrix<>)` | Asynchronously generates predictions for new data points using the trained M5 model tree. |
| `PredictSingle(Vector<>)` | Predicts a value for a single input vector by traversing the tree. |
| `PruneTreeAsync(DecisionTreeNode<>)` | Asynchronously prunes the tree to reduce complexity and prevent overfitting. |
| `Serialize` | Serializes the M5 model tree to a byte array, including linear models at leaf nodes. |
| `SerializeM5Node(BinaryWriter,DecisionTreeNode<>)` | Serializes an M5 tree node including its linear model if present. |
| `SplitData(Matrix<>,Vector<>,Int32,)` | Splits the data into left and right subsets based on a feature and threshold. |
| `TrainAsync(Matrix<>,Vector<>)` | Asynchronously trains the M5 model tree using the provided features and target values. |
| `VarianceFromMoments(,,Int32)` | Computes the sample variance of a group of target values from its running sum and sum-of-squares, used by the standard-deviation-reduction split search. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | The configuration options for the M5 model tree algorithm. |

