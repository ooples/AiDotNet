---
title: "DecisionTreeClassifier<T>"
description: "A decision tree classifier that learns a hierarchy of decision rules from training data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Trees`

A decision tree classifier that learns a hierarchy of decision rules from training data.

## For Beginners

Imagine playing a game of "20 Questions" to classify things. The decision tree learns
which questions (based on features) best separate the different classes.

Example: Classifying whether to play tennis

1. Is it raining? -> No: Go to step 2, Yes: Don't play
2. Is humidity > 75%? -> No: Play!, Yes: Don't play

Each question splits the data based on a feature value, and leaves contain the final decisions.

## How It Works

Decision trees are non-parametric supervised learning algorithms that learn decision rules
inferred from data features. They partition the feature space into regions and assign
class labels to each region.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DecisionTreeClassifier(DecisionTreeClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the DecisionTreeClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureImportances` |  |
| `LeafCount` |  |
| `MaxDepth` |  |
| `NodeCount` |  |
| `Options` | Gets the decision tree specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `BuildTree(Matrix<>,Vector<>,List<Int32>,Int32)` | Builds the decision tree recursively. |
| `CalculateActualDepth(DecisionNode<>)` | Calculates the actual depth of the tree. |
| `CalculateEntropy(Int32[],Int32)` | Calculates entropy. |
| `CalculateGiniImpurity(Int32[],Int32)` | Calculates Gini impurity. |
| `CalculateImpurity(Vector<>,List<Int32>)` | Calculates impurity for a set of samples based on the configured criterion. |
| `Clone` |  |
| `CloneNode(DecisionNode<>)` | Deep clones a decision tree node. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `CountLeaves(DecisionNode<>)` | Counts the number of leaf nodes. |
| `CountNodes(DecisionNode<>)` | Counts the total number of nodes. |
| `CreateLeafNode(Vector<>,List<Int32>)` | Creates a leaf node with class probabilities. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `DeserializeNode(JObject)` | Deserializes a decision node from a JSON object. |
| `FindBestSplit(Matrix<>,Vector<>,List<Int32>)` | Finds the best split for the given indices. |
| `GetClassIndex()` | Gets the class index for a label. |
| `GetFeaturesToConsider` | Gets the features to consider for splitting based on MaxFeatures setting. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `NormalizeFeatureImportances` | Normalizes feature importances to sum to 1. |
| `PredictProbabilities(Matrix<>)` |  |
| `Serialize` |  |
| `SerializeNode(DecisionNode<>)` | Serializes a decision node to a dictionary for JSON serialization. |
| `SetParameters(Vector<>)` |  |
| `ShouldStop(List<Int32>,Vector<>,Int32)` | Determines if we should stop splitting. |
| `SplitData(Matrix<>,List<Int32>,Int32,)` | Splits data based on a feature and threshold. |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |
| `TraverseTree(Vector<>,DecisionNode<>)` | Traverses the tree to find the appropriate leaf node. |
| `UpdateFeatureImportances(Int32,,Int32)` | Updates feature importances based on the gain from a split. |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_random` | Random number generator for feature selection when MaxFeatures is set. |
| `_root` | The root node of the decision tree. |

