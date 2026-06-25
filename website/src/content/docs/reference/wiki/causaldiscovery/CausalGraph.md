---
title: "CausalGraph<T>"
description: "Represents a causal Directed Acyclic Graph (DAG) discovered from observational data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery`

Represents a causal Directed Acyclic Graph (DAG) discovered from observational data.
This is both a standalone queryable model and an analysis result.

## For Beginners

Think of this as a map of cause-and-effect relationships. Each variable
is a node, and arrows between nodes show which variables directly cause changes in others.
The weight of each arrow tells you how strong the causal effect is.

You can query this graph to answer questions like:

- "What causes variable X?" — use `Int32)`
- "What does variable X affect?" — use `Int32)`
- "What is the Markov blanket of X?" — use `Int32)`
- "What order should I process variables?" — use `TopologicalSort`

## How It Works

A causal graph encodes the causal relationships between variables as a weighted adjacency matrix.
An edge from variable i to variable j with weight w means "variable i directly causes variable j
with strength w." The graph is guaranteed to be a DAG (no cycles), meaning you cannot follow
directed edges and return to the starting variable.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CausalGraph(Matrix<>,String[])` | Initializes a new CausalGraph from a weighted adjacency matrix. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdjacencyMatrix` | Gets the weighted adjacency matrix where entry [i,j] represents the causal effect from variable i to variable j. |
| `Density` | Gets the density of the graph (fraction of possible edges that exist). |
| `EdgeCount` | Gets the total number of edges in the graph. |
| `FeatureNames` | Gets the feature/variable names. |
| `NumVariables` | Gets the number of variables in the graph. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeInterventionalDistribution(Int32,,Int32,Matrix<>)` | Computes the interventional distribution P(target \| do(intervention = value)) using the truncated factorization formula from Pearl's do-calculus. |
| `ComputeInterventionalDistribution(String,,String,Matrix<>)` | Computes the interventional distribution using named variables. |
| `ComputeInterventionalFallback(Int32,,Int32,Matrix<>)` | Fallback for computing interventional distribution when the graph is not a DAG. |
| `GetAncestors(Int32)` | Gets all ancestor variables (direct and indirect causes) of the specified variable. |
| `GetChildren(Int32)` | Gets the indices of child variables (direct effects) of the specified variable. |
| `GetChildren(String)` | Gets the child variables (direct effects) of a named variable. |
| `GetDescendants(Int32)` | Gets all descendant variables (direct and indirect effects) of the specified variable. |
| `GetEdgeWeight(Int32,Int32)` | Gets the weight of the edge from one variable to another. |
| `GetEdgeWeight(String,String)` | Gets the weight of the edge between named variables. |
| `GetEdges(Double)` | Gets all edges in the graph as (from, to, weight) tuples, optionally filtered by minimum weight. |
| `GetMarkovBlanket(Int32)` | Gets the Markov blanket of a variable: its parents, children, and co-parents of its children. |
| `GetNamedEdges(Double)` | Gets all edges as named tuples for human-readable output. |
| `GetNodeImportance` | Computes a simple node importance score based on out-degree weighted by edge strength. |
| `GetParents(Int32)` | Gets the indices of parent variables (direct causes) of the specified variable. |
| `GetParents(String)` | Gets the parent variables (direct causes) of a named variable. |
| `HasEdge(Int32,Int32)` | Returns whether there is a directed edge from one variable to another. |
| `IsDAG` | Returns whether the graph is a valid DAG (no directed cycles). |
| `TopologicalSort` | Returns a topological ordering of the variables (causes before effects). |

