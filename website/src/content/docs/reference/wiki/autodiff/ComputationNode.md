---
title: "ComputationNode<T>"
description: "Represents a node in the automatic differentiation computation graph."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Autodiff`

Represents a node in the automatic differentiation computation graph.

## For Beginners

This represents a single step in a calculation that can be differentiated.

Think of it like this:

- A node stores a value (like the output of adding two numbers)
- It remembers what inputs were used to create this value (the two numbers)
- It knows how to calculate gradients (derivatives) with respect to its inputs
- Connecting nodes together forms a graph that tracks the entire calculation

This enables automatic differentiation, where gradients can be computed
automatically for complex operations by chaining together simple derivatives.

## How It Works

A ComputationNode is a fundamental building block of automatic differentiation.
It represents a value in a computational graph, along with information about
how to compute gradients with respect to that value. Each node stores its value,
gradient, parent nodes (inputs), and a backward function for gradient computation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ComputationNode(Tensor<>,Boolean,List<ComputationNode<>>,Action<Tensor<>>,String)` | Initializes a new instance of the `ComputationNode` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BackwardFunction` | Gets or sets the backward function that computes gradients for parent nodes. |
| `Gradient` | Gets or sets the gradient accumulated at this node. |
| `Name` | Gets or sets an optional name for this node (useful for debugging). |
| `OperationParams` | Gets or sets additional operation-specific parameters (used for JIT compilation). |
| `OperationType` | Gets or sets the type of operation that created this node (used for JIT compilation). |
| `Parents` | Gets or sets the parent nodes (inputs) that were used to compute this node's value. |
| `RequiresGradient` | Gets or sets a value indicating whether this node requires gradient computation. |
| `Value` | Gets or sets the value stored in this node. |

## Methods

| Method | Summary |
|:-----|:--------|
| `TopologicalSort` | Performs a topological sort of the computation graph rooted at this node. |
| `ZeroGradient` | Zeros out the gradient for this node. |
| `ZeroGradientRecursive` | Recursively zeros out gradients for this node and all its ancestors. |

