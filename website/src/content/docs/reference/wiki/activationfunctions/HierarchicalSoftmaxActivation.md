---
title: "HierarchicalSoftmaxActivation<T>"
description: "Implements the Hierarchical Softmax activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Hierarchical Softmax activation function for neural networks.

## For Beginners

Hierarchical Softmax is an efficient alternative to the standard Softmax function,
especially when dealing with a large number of output classes (like thousands of words in language models).

While regular Softmax calculates probabilities for all possible classes at once (which can be slow
with many classes), Hierarchical Softmax organizes classes in a tree structure:

- Think of it like a "20 Questions" game where each question narrows down the possibilities
- Each node in the tree represents a binary decision (left or right)
- The final probability is calculated by multiplying probabilities along the path to a class

This approach reduces computation from O(N) to O(log N), where N is the number of classes,
making it much faster for problems with many output classes.

Common uses include:

- Natural language processing (predicting words from vocabularies)
- Classification problems with many categories
- Any task where computing standard Softmax would be too slow

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HierarchicalSoftmaxActivation(Int32)` | Initializes a new instance of the Hierarchical Softmax activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NodeWeightsTensor` | Gets the node weights as a tensor for use in computation graphs. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate(Tensor<>)` | Applies HierarchicalSoftmax to a tensor via engine primitives so the gradient tape records every step. |
| `Activate(Vector<>)` | Applies the Hierarchical Softmax activation function to transform input vectors into class probabilities. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `ApplyToGraph(ComputationNode<>,ComputationNode<>)` | Applies Hierarchical Softmax with externally provided weights for full training support. |
| `ComputePathDerivative(Vector<>,Int32)` | Calculates the derivative of the path probability for a specific class. |
| `ComputePathProbability(Vector<>,Int32)` | Computes the probability of a specific class by traversing the binary tree. |
| `Derivative(Vector<>)` | Calculates the derivative (gradient) of the Hierarchical Softmax function. |
| `InitializeWeights` | Initializes the weights for all nodes in the binary tree with small random values. |
| `SupportsScalarOperations` | Indicates whether this activation function can operate on individual scalar values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_nodeWeights` | The weights for each node in the binary tree. |
| `_numClasses` | The total number of output classes. |
| `_treeDepth` | The depth of the binary tree used to represent the hierarchical structure. |
| `_weightInputDim` | The input dimension that the weight matrix was initialized for. |

