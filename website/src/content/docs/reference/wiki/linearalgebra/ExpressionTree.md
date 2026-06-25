---
title: "ExpressionTree<T, TInput, TOutput>"
description: "Represents a symbolic expression tree for mathematical operations that can be used for symbolic regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LinearAlgebra`

Represents a symbolic expression tree for mathematical operations that can be used for symbolic regression.

## How It Works

**For Beginners:** An ExpressionTree is like a mathematical formula represented as a tree structure.
Each node in the tree is either a number, a variable, or an operation (like addition or multiplication).
This allows the AI to create and evolve mathematical formulas that can model your data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExpressionTree(ExpressionNodeType,,ExpressionTree<,,>,ExpressionTree<,,>,ILossFunction<>)` | Creates a new expression tree node with the specified properties. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Coefficients` | Gets a vector containing all coefficient values in this expression tree. |
| `Complexity` | Gets the complexity of this expression tree, measured as the total number of nodes. |
| `DefaultLossFunction` | Gets the default loss function used by this model for gradient computation. |
| `FeatureCount` | Gets the number of features (variables) used in this expression tree. |
| `Id` | Gets the unique identifier for this node. |
| `Left` | Gets the left child node of this node. |
| `ParameterCount` | Gets the number of parameters (constant nodes) in this expression tree. |
| `Parent` | Gets the parent node of this node. |
| `RequiredFeatureCount` | Gets the minimum number of features required for input data to this expression tree. |
| `Right` | Gets the right child node of this node. |
| `Type` | Gets the type of this node (constant, variable, or operation). |
| `Value` | Gets the value stored in this node. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies pre-computed gradients to update the model parameters (constants in the expression tree). |
| `CalculateFeatureCount` | Calculates the number of unique features used in this expression tree. |
| `CalculateRequiredFeatureCount` | Calculates the minimum number of features required for input data. |
| `CollectNodes(ExpressionTree<,,>,List<ExpressionTree<,,>>)` | Helper method that recursively collects all nodes in the tree. |
| `CollectUniqueFeatures(ExpressionTree<,,>,HashSet<Int32>)` | Recursively collects unique feature indices used in a node and its children. |
| `ComputeGradients(,,ILossFunction<>)` | Computes gradients of the loss function with respect to model parameters WITHOUT updating parameters. |
| `Copy` | Creates a deep copy of this expression tree. |
| `Crossover(IFullModel<,,>,Double)` | Combines this expression tree with another to create a new "offspring" expression tree. |
| `DeepCopy` | Creates a deep copy of this expression tree. |
| `Deserialize(BinaryReader)` | Deserializes an expression tree from a binary reader. |
| `Deserialize(Byte[])` | Loads an expression tree from a byte array, replacing the current tree's structure. |
| `Evaluate(Vector<>)` | Evaluates this expression tree for a given input vector. |
| `FindMaxFeatureIndex(ExpressionTree<,,>,Int32)` | Recursively finds the maximum feature index used in a node and its children. |
| `FindNodeById(Int32)` | Finds a node in the tree by its unique identifier. |
| `Fit(Matrix<>,Vector<>)` | Fits the expression tree to the provided training data. |
| `GenerateRandomTree(Int32)` | Creates a random expression tree with a specified maximum depth. |
| `GetActiveFeatureIndices` | Gets the indices of all features (variables) used in this expression tree. |
| `GetAllNodes` | Gets a list of all nodes in this expression tree. |
| `GetFeatureImportance` | Gets the feature importance scores for this expression tree. |
| `GetModelMetadata` | Gets metadata about this expression tree model. |
| `GetParameters` | Gets the parameters of this expression tree. |
| `IsFeatureUsed(Int32)` | Checks if a specific feature (variable) is used in this expression tree. |
| `IsFeatureUsedRecursive(ExpressionTree<,,>,Int32)` | Recursively checks if a specific feature is used in a node or its children. |
| `LoadModel(String)` | Loads an expression tree model from a file. |
| `LoadState(Stream)` | Loads the expression tree's state (structure and values) from a stream. |
| `Mutate(Double)` | Creates a modified version of this expression tree by applying random mutations. |
| `Predict()` | Makes a prediction for an input example. |
| `Predict(Matrix<>)` | Makes predictions using this expression tree for multiple input samples. |
| `PredictMatrix(Matrix<>)` | Makes predictions for all rows in a matrix. |
| `PredictTensor(Tensor<>)` | Makes predictions for all samples in a tensor. |
| `ReplaceRandomSubtree(ExpressionTree<,,>,ExpressionTree<,,>)` | Replaces a random subtree in the given tree with the provided replacement subtree. |
| `SaveModel(String)` | Saves the expression tree model to a file. |
| `SaveState(Stream)` | Saves the expression tree's current state (structure and values) to a stream. |
| `SelectRandomSubtree(ExpressionTree<,,>)` | Selects a random subtree from the given expression tree. |
| `Serialize` | Converts this expression tree to a byte array for storage or transmission. |
| `Serialize(BinaryWriter)` | Writes this expression tree to a binary stream. |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` | Sets the active feature indices for this expression tree. |
| `SetLeft(ExpressionTree<,,>)` | Sets the left child of this node and updates the parent reference of the child. |
| `SetParameters(Vector<>)` | Sets the parameters (constant values) of this expression tree, modifying it in place. |
| `SetRight(ExpressionTree<,,>)` | Sets the right child of this node and updates the parent reference of the child. |
| `SetType(ExpressionNodeType)` | Sets the type of this node. |
| `SetValue()` | Sets the value of this node. |
| `ToString` | Returns a string representation of this expression tree. |
| `Train(,)` | Trains the expression tree on a single input-output pair. |
| `Train(Matrix<>,Vector<>)` | Trains the expression tree on the provided data. |
| `UpdateCoefficients(Vector<>)` | Creates a new expression tree with updated coefficient values. |
| `ValidateMatrixFeatures(Matrix<>)` | Validates that a matrix has compatible features for this expression tree. |
| `ValidateTensorFeatures(Tensor<>)` | Validates that a tensor has compatible features for this expression tree. |
| `ValidateVectorFeatures(Vector<>)` | Validates that a vector has compatible features for this expression tree. |
| `WithParameters(Vector<>)` | Creates a new expression tree with updated parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_defaultLossFunction` | The default loss function used by this model for gradient computation. |
| `_featureCount` | Cached count of features used in this expression tree. |
| `_nextId` | Static counter used to generate unique IDs for expression tree nodes. |
| `_random` | Shared random number generator for all mutation and crossover operations. |
| `_requiredFeatureCount` | Cached required feature count (max feature index + 1) for validation. |

