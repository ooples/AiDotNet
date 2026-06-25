---
title: "NodeClassificationTask<T>"
description: "Represents a node classification task where the goal is to predict labels for individual nodes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Structures`

Represents a node classification task where the goal is to predict labels for individual nodes.

## For Beginners

Node classification is like categorizing people in a social network.

**Real-world examples:**

**Social Networks:**

- Nodes: Users
- Task: Predict user interests/communities
- How: Use profile features + friend connections
- Example: "Is this user interested in sports?"

**Citation Networks:**

- Nodes: Research papers
- Task: Classify paper topics
- How: Use paper abstracts + citation links
- Example: Papers citing each other often share topics

**Fraud Detection:**

- Nodes: Financial accounts
- Task: Detect fraudulent accounts
- How: Use transaction patterns + account relationships
- Example: Fraudsters often form connected clusters

**Key Insight:** Node classification leverages the graph structure. Connected nodes often
share similar properties (homophily), so a node's neighbors provide valuable information
for prediction.

## How It Works

Node classification is a fundamental graph learning task where each node in a graph has a label,
and the goal is to predict labels for unlabeled nodes based on:

- Node features
- Graph structure (connections between nodes)
- Labels of neighboring nodes

## Properties

| Property | Summary |
|:-----|:--------|
| `Graph` | The graph data containing nodes, edges, and features. |
| `IsMultiLabel` | Whether this is a multi-label classification task. |
| `Labels` | Node labels for all nodes in the graph. |
| `NumClasses` | Number of classes in the classification task. |
| `TestIndices` | Indices of nodes to use for testing. |
| `TrainIndices` | Indices of nodes to use for training. |
| `ValIndices` | Indices of nodes to use for validation. |

