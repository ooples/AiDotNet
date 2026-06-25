---
title: "TreeIndividual"
description: "Represents an individual in genetic programming with a tree structure."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Genetics`

Represents an individual in genetic programming with a tree structure.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TreeIndividual(NodeGene)` | Creates a tree individual with the specified root node. |
| `TreeIndividual(Random,List<String>,Boolean)` | Creates a new tree individual with a random tree. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CollectNodes(NodeGene,List<NodeGene>)` | Collects all nodes in the tree. |
| `EnsureValidDepth` | Ensures the tree doesn't exceed the maximum allowed depth. |
| `Evaluate(Dictionary<String,Double>)` | Evaluates the tree for a given input and returns the result. |
| `EvaluateNode(NodeGene,Dictionary<String,Double>)` | Recursively evaluates a node in the tree. |
| `FindNodeDepth(NodeGene,NodeGene,Int32)` | Recursively finds the depth of a target node in the tree. |
| `GenerateRandomTree(Int32,List<String>,Boolean,Nullable<Int32>)` | Generates a random subtree. |
| `GetDepth` | Gets the depth of the tree. |
| `GetExpression` | Gets a string representation of the tree. |
| `GetFunctionArity(String)` | Gets the arity (number of arguments) for a function. |
| `GetNodeDepth(NodeGene)` | Gets the depth of a node. |
| `GetNodeDepthInTree(NodeGene)` | Gets the depth of a specific node within the tree. |
| `NodeToString(NodeGene)` | Converts a node to string representation. |
| `PermutationMutation` | Performs permutation mutation by randomizing the order of arguments. |
| `PointMutation` | Performs point mutation on the tree by changing a random terminal or function. |
| `ReplaceInNode(NodeGene,NodeGene,NodeGene)` | Recursively searches and replaces a node in the tree. |
| `ReplaceSubtree(NodeGene,NodeGene)` | Replaces a subtree with another subtree. |
| `SelectRandomNode` | Selects a random node from the tree. |
| `SubtreeMutation` | Performs subtree mutation by replacing a random subtree with a new one. |

