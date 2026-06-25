---
title: "AsyncDecisionTreeRegressionBase<T>"
description: "Represents an abstract base class for asynchronous decision tree regression models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Regression`

Represents an abstract base class for asynchronous decision tree regression models.

## For Beginners

A decision tree is a type of machine learning model that makes predictions
by following a series of yes/no questions about the input data. It's like a flowchart that helps the
computer decide what prediction to make.

For example, if you're trying to predict if it will rain:

- Is the humidity high? If yes, go to next question. If no, predict no rain.
- Are there clouds? If yes, predict rain. If no, predict no rain.

This class provides the basic structure for building these types of models, but with more complex
questions and answers based on numerical data.

## How It Works

This class provides a foundation for implementing decision tree regression models that can be trained
and used for predictions asynchronously. It includes methods for training, prediction, serialization,
and deserialization of the model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AsyncDecisionTreeRegressionBase(DecisionTreeOptions,IRegularization<,Matrix<>,Vector<>>,ILossFunction<>)` | Initializes a new instance of the AsyncDecisionTreeRegressionBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |
| `Engine` | Gets the global execution engine for vector operations. |
| `FeatureImportances` | Gets or sets the importance of each feature in making predictions. |
| `FeatureNames` | Gets or sets the feature names. |
| `MaxDepth` | Gets the maximum depth of the decision tree. |
| `NumberOfTrees` | Gets the number of trees in the model. |
| `Options` | Gets the options used to configure the decision tree. |
| `Regularization` | Gets the regularization method used to prevent overfitting. |
| `SoftTreeTemperature` | Gets or sets the temperature parameter for soft decision tree mode. |
| `UseSoftTree` | Gets or sets whether to use soft (differentiable) tree mode for JIT compilation support. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `CalculateFeatureImportancesAsync(Int32)` | Asynchronously calculates the importance of each feature in the model. |
| `Clone` | Creates a clone of the decision tree model. |
| `CollectActiveFeatures(DecisionTreeNode<>,HashSet<Int32>)` | Collects all feature indices used in the tree. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `CountNodes(DecisionTreeNode<>)` | Counts the total number of nodes in the tree. |
| `CreateNewInstance` | Creates a new instance of the async decision tree model with the same options. |
| `DeepCloneNode(DecisionTreeNode<>)` | Creates a deep clone of a node and its children. |
| `DeepCopy` | Creates a deep copy of the decision tree model. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `DeserializeNode(BinaryReader)` | Deserializes a single node of the decision tree. |
| `DeserializeNodeFromVector(Vector<>,Int32)` | Deserializes a node and its children from a parameter vector. |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources held by this async decision-tree regressor. |
| `GetActiveFeatureIndices` | Gets the indices of all features that are used in the decision tree. |
| `GetDynamicShapeInfo` |  |
| `GetFeatureImportance` | Gets the feature importance scores as a dictionary. |
| `GetInputShape` | Saves the model to a file. |
| `GetModelMetadata` | Gets metadata about the trained model. |
| `GetOptions` |  |
| `GetOutputShape` |  |
| `GetParameters` | Gets the model parameters as a vector representation. |
| `IsFeatureUsed(Int32)` | Determines whether a specific feature is used in the decision tree. |
| `IsFeatureUsedInSubtree(DecisionTreeNode<>,Int32)` | Checks if a specific feature is used in a subtree. |
| `LoadModel(String)` | Loads the model from a file. |
| `LoadState(Stream)` | Loads the model's state from a stream. |
| `Predict(Matrix<>)` | Makes predictions using the trained model. |
| `PredictAsync(Matrix<>)` | Asynchronously makes predictions using the trained model. |
| `SanitizeParameters(Vector<>)` |  |
| `SaveState(Stream)` | Saves the model's current state to a stream. |
| `Serialize` | Serializes the model to a byte array. |
| `SerializeNode(BinaryWriter,DecisionTreeNode<>)` | Serializes a single node of the decision tree. |
| `SerializeNodeToVector(DecisionTreeNode<>,Vector<>,Int32)` | Serializes a node and its children to a parameter vector. |
| `Train(Matrix<>,Vector<>)` | Trains the decision tree model on the provided data. |
| `TrainAsync(Matrix<>,Vector<>)` | Asynchronously trains the decision tree model on the provided data. |
| `WithParameters(Vector<>)` | Creates a new instance of the model with the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Gets the numeric operations for the type T. |
| `Root` | Gets or sets the root node of the decision tree. |
| `_defaultLossFunction` | Gets the default loss function for this async tree-based regression model. |
| `_random` | Random number generator used for tree building and sampling. |

