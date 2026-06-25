---
title: "DecisionTreeRegression<T>"
description: "Represents a decision tree regression model that predicts continuous values based on input features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Represents a decision tree regression model that predicts continuous values based on input features.

## For Beginners

A decision tree regression is like a flowchart that helps predict numerical values.

Think of it like answering a series of yes/no questions to reach a prediction:

- "Is the temperature above 75—F?"
- "Is the humidity below 50%?"
- "Is it a weekend?"

Each question splits the data into two groups, and the tree learns which questions to ask 
to make the most accurate predictions. For example, a decision tree might predict house prices 
based on features like square footage, number of bedrooms, and neighborhood.

The model is called a "tree" because it resembles an upside-down tree, with a single starting point (root) 
that branches out into multiple endpoints (leaves) where the final predictions are made.

## How It Works

Decision tree regression builds a model in the form of a tree structure where each internal node represents a 
decision based on a feature, each branch represents an outcome of that decision, and each leaf node 
represents a predicted value. The model is trained by recursively splitting the data based on the optimal 
feature and threshold that minimizes the prediction error.

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
    .ConfigureModel(new DecisionTreeRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained DecisionTreeRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DecisionTreeRegression(DecisionTreeOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the `DecisionTreeRegression` class with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumberOfTrees` | Gets the number of trees in this model, which is always 1 for a single decision tree. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildTree(Matrix<>,Vector<>,Int32)` | Builds the decision tree recursively. |
| `BuildTreeFast(Matrix<>,Vector<>)` | Allocation-light variance-reduction tree builder. |
| `BuildTreeWithWeights(DecisionTreeNode<>,Matrix<>,Vector<>,Vector<>,Int32)` | Builds a decision tree using weighted samples. |
| `CalculateFeatureImportances(Int32)` | Calculates feature importances based on the number of features. |
| `CalculateFeatureImportances(Matrix<>)` | Calculates the importance scores for all features based on their contribution to the tree. |
| `CalculateFeatureImportancesRecursive(DecisionTreeNode<>,Int32)` | Recursively calculates feature importances by traversing the tree. |
| `CalculateNodeImportance(DecisionTreeNode<>)` | Calculates the importance of a single node based on the variance reduction it achieves. |
| `CalculateWeightedLeafValue(Vector<>,Vector<>)` | Calculates the weighted prediction value for a leaf node. |
| `CalculateWeightedVarianceReduction(Vector<>,Vector<>)` | Calculates the weighted variance reduction for a set of target values and weights. |
| `CreateNewInstance` | Creates a new instance of the decision tree regression model with the same options. |
| `Deserialize(Byte[])` | Loads a previously serialized decision tree model from a byte array. |
| `DeserializeNode(BinaryReader)` | Deserializes a tree node from a binary reader. |
| `FindBestSplitFast(Double[][],Double[],Int32[],List<Int32>)` | Index-subset variance-reduction split search in native double: each candidate feature is sorted once, then a single sweep with running left-partition sums (count, Σy, Σy²) scores every threshold in O(1). |
| `FindBestSplitWithWeights(Matrix<>,Vector<>,Vector<>,IEnumerable<Int32>)` | Finds the best feature and threshold to split the data based on weighted samples. |
| `GetFeatureImportance(Int32)` | Gets the importance score of a specific feature in the decision tree model. |
| `GetModelMetadata` | Gets metadata about the decision tree model and its configuration. |
| `GetOptions` |  |
| `NormalizeFeatureImportances` | Normalizes feature importance scores to sum to 1. |
| `Predict(Matrix<>)` | Predicts target values for the provided input features using the trained decision tree model. |
| `PredictSingle(Vector<>,DecisionTreeNode<>)` | Predicts the target value for a single sample by traversing the decision tree. |
| `Serialize` | Serializes the decision tree model to a byte array for storage or transmission. |
| `SerializeNode(DecisionTreeNode<>,BinaryWriter)` | Serializes a tree node to a binary writer. |
| `SplitDataWithWeights(Matrix<>,Vector<>,Vector<>,Int32,)` | Splits the data into left and right subsets based on a feature and threshold. |
| `Train(Matrix<>,Vector<>)` | Trains the decision tree model using the provided input features and target values. |
| `TrainWithWeights(Matrix<>,Vector<>,Vector<>)` | Trains the decision tree model using the provided input features, target values, and sample weights. |
| `WeightedSseFromMoments(,,)` | Weighted sum of squared deviations of a group from its running moments: Σw·y² − (Σw·y)²/Σw, which equals (group weight) × (group weighted variance). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_featureImportances` | Vector storing the importance scores for each feature in the model. |
| `_options` | The configuration options for the decision tree algorithm. |
| `_random` | Random number generator used for feature selection and other randomized operations. |
| `_regularization` | The regularization strategy applied to the model to prevent overfitting. |

