---
title: "GraphClassificationTask<T>"
description: "Represents a graph classification task where the goal is to classify entire graphs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Structures`

Represents a graph classification task where the goal is to classify entire graphs.

## For Beginners

Graph classification is like determining the category of a complex object.

**Real-world examples:**

**Molecular Property Prediction:**

- Input: Molecular graph (atoms as nodes, bonds as edges)
- Task: Predict molecular properties
- Examples:
* Is this molecule toxic?
* What is the solubility?
* Will this be a good drug candidate?
- Dataset: ZINC, QM9, BACE

**Protein Function Prediction:**

- Input: Protein structure graph
- Task: Predict protein function or family
- How: Analyze amino acid sequences and 3D structure

**Chemical Reaction Prediction:**

- Input: Reaction graph showing reactants and products
- Task: Predict reaction type or outcome

**Social Network Analysis:**

- Input: Community subgraphs
- Task: Classify community type or behavior
- Example: Identify bot networks vs organic communities

**Code Analysis:**

- Input: Abstract syntax tree (AST) or control flow graph
- Task: Detect bugs, classify code functionality
- Example: "Is this code snippet vulnerable to SQL injection?"

**Key Challenge:** Graph-level representation

- Must aggregate information from all nodes and edges
- Common approaches: Global pooling, hierarchical pooling, set2set

## How It Works

Graph classification assigns a label to an entire graph based on its structure and node/edge features.
Unlike node classification (classify individual nodes) or link prediction (predict edges),
graph classification treats the whole graph as a single data point.

## Properties

| Property | Summary |
|:-----|:--------|
| `AvgNumEdges` | Average number of edges per graph (for informational purposes). |
| `AvgNumNodes` | Average number of nodes per graph (for informational purposes). |
| `IsMultiLabel` | Whether this is a multi-label classification task. |
| `IsRegression` | Whether this is a regression task instead of classification. |
| `NumClasses` | Number of classes in the classification task. |
| `TestGraphs` | List of test graphs. |
| `TestLabels` | Labels for test graphs. |
| `TrainGraphs` | List of training graphs. |
| `TrainLabels` | Labels for training graphs. |
| `ValGraphs` | List of validation graphs. |
| `ValLabels` | Labels for validation graphs. |

