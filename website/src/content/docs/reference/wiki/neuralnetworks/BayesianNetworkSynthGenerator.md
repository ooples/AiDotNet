---
title: "BayesianNetworkSynthGenerator<T>"
description: "Bayesian Network Synthesis generator that learns a DAG structure over features, estimates conditional probability tables (CPTs), and generates synthetic data via ancestral sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

Bayesian Network Synthesis generator that learns a DAG structure over features,
estimates conditional probability tables (CPTs), and generates synthetic data
via ancestral sampling.

## For Beginners

Think of this as building a probabilistic "family tree" of your features:

Step 1: Figure out which features depend on which others (the DAG)
Step 2: For each feature, learn "if parent features have values X, this feature is Y with probability Z"
Step 3: To generate a new row, start with features that have no parents and work downward

Advantages: Fast, interpretable, no GPU needed.
Disadvantage: Less flexible than deep learning for complex distributions.

## How It Works

This is a classical statistical approach (no neural networks):

1. Discretize continuous features into bins
2. Learn a DAG structure using greedy hill-climbing with BIC scoring
3. Estimate CPTs using maximum likelihood with Laplace smoothing
4. Generate data by sampling from root nodes to leaf nodes in topological order

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BayesianNetworkSynthGenerator(BayesianNetworkSynthOptions<>)` | Initializes a new instance of the `BayesianNetworkSynthGenerator` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AncestralSample` | Generates a single sample using ancestral sampling (topological order). |
| `ComputeLocalBIC(Int32[][],Int32,List<Int32>,Int32,Int32,Double)` | Computes the BIC score for a node given its parent set. |
| `DiscretizeData(Matrix<>)` | Discretizes continuous data into bins using equal-width binning. |
| `EstimateCPTs(Int32[][])` | Estimates conditional probability tables (CPTs) for each feature given its parents. |
| `FitInternal(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` |  |
| `GenerateInternal(Int32,Vector<>,Vector<>)` |  |
| `GetParentKey(Int32[],List<Int32>)` | Creates a string key from the parent values of a data row. |
| `LearnStructure(Int32[][])` | Learns the DAG structure using greedy hill-climbing with BIC scoring. |
| `SampleFromDistribution(Double[])` | Samples a bin index from a discrete probability distribution. |
| `TopologicalSort` | Computes a topological ordering of the DAG nodes. |
| `WouldCreateCycle(List<Int32>[],Int32,Int32)` | Checks if adding an edge from parent to child would create a cycle in the DAG. |

