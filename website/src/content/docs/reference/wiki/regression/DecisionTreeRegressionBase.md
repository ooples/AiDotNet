---
title: "DecisionTreeRegressionBase<T>"
description: "Provides a base implementation for decision tree regression models that predict continuous values."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Regression`

Provides a base implementation for decision tree regression models that predict continuous values.

## For Beginners

This is a template for creating decision tree models that predict numerical values.

A decision tree works like a flowchart of yes/no questions to make predictions:

- Start at the top (root) of the tree
- At each step, answer a question about your data
- Follow the appropriate path based on your answer
- Continue until you reach an endpoint that provides a prediction

This base class provides the common structure and behaviors that all decision tree models share,
while allowing specific implementations to customize how the tree is built and used.

Think of it like a blueprint for building different types of decision trees, where specific 
implementations can fill in the details according to their requirements.

## How It Works

This abstract class implements common functionality for decision tree regression models, providing a framework
for building predictive models based on decision trees. It manages the tree structure, handles serialization
and deserialization, and defines the interface that concrete implementations must support.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DecisionTreeRegressionBase(DecisionTreeOptions,IRegularization<,Matrix<>,Vector<>>,ILossFunction<>)` | Initializes a new instance of the `DecisionTreeRegressionBase` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |
| `Engine` | Gets the global execution engine for vector operations. |
| `FeatureImportances` | Gets the importance scores for each feature used in the model. |
| `FeatureNames` | Gets or sets the feature names. |
| `MaxDepth` | Gets the maximum depth of the decision tree. |
| `NumberOfTrees` | Gets the number of trees in this model, which is always 1 for a single decision tree. |
| `Options` | Gets the configuration options used by the decision tree algorithm. |
| `Regularization` | Gets the regularization strategy applied to the model to prevent overfitting. |
| `SoftTreeTemperature` | Gets or sets the temperature parameter for soft decision tree mode. |
| `UseSoftTree` | Gets or sets whether to use soft (differentiable) tree mode for JIT compilation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `CalculateFeatureImportances(Int32)` | Calculates the importance scores for all features used in the model. |
| `Clone` | Creates a clone of the decision tree model. |
| `CollectActiveFeatures(DecisionTreeNode<>,HashSet<Int32>)` | Collects all feature indices used in the tree. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `CountNodes(DecisionTreeNode<>)` | Counts the total number of nodes in the tree. |
| `CreateNewInstance` | Creates a new instance of the decision tree model with the same options. |
| `DeepCloneNode(DecisionTreeNode<>)` | Creates a deep clone of a node and its children. |
| `DeepCopy` | Creates a deep copy of the decision tree model. |
| `Deserialize(Byte[])` | Loads a previously serialized decision tree model from a byte array. |
| `DeserializeNode(BinaryReader)` | Deserializes a tree node from a binary reader. |
| `DeserializeNodeFromVector(Vector<>,Int32)` | Deserializes a node and its children from a parameter vector. |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources held by this decision-tree regressor. |
| `GetActiveFeatureIndices` | Gets the indices of all features that are used in the decision tree. |
| `GetDynamicShapeInfo` |  |
| `GetFeatureImportance` | Gets the feature importance scores as a dictionary. |
| `GetInputShape` |  |
| `GetModelMetadata` | Gets metadata about the decision tree model and its configuration. |
| `GetOptions` |  |
| `GetOutputShape` |  |
| `GetParameters` | Gets the model parameters as a vector representation. |
| `IsFeatureUsed(Int32)` | Determines whether a specific feature is used in the decision tree. |
| `IsFeatureUsedInSubtree(DecisionTreeNode<>,Int32)` | Checks if a specific feature is used in a subtree. |
| `LoadState(Stream)` | Loads the model's state from a stream. |
| `Predict(Matrix<>)` | Predicts target values for the provided input features using the trained decision tree model. |
| `SanitizeParameters(Vector<>)` |  |
| `SaveState(Stream)` | Saves the model's current state to a stream. |
| `Serialize` | Serializes the decision tree model to a byte array for storage or transmission. |
| `SerializeNode(BinaryWriter,DecisionTreeNode<>)` | Serializes a tree node to a binary writer. |
| `SerializeNodeToVector(DecisionTreeNode<>,Vector<>,Int32)` | Serializes a node and its children to a parameter vector. |
| `Train(Matrix<>,Vector<>)` | Trains the decision tree model using the provided input features and target values. |
| `WithParameters(Vector<>)` | Creates a new instance of the model with the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides operations for performing numeric calculations appropriate for the type T. |
| `Root` | The root node of the decision tree. |
| `_defaultLossFunction` | Gets the default loss function for this tree-based regression model. |

