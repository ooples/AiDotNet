---
title: "LinkPredictionTask<T>"
description: "Represents a link prediction task where the goal is to predict missing or future edges."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Structures`

Represents a link prediction task where the goal is to predict missing or future edges.

## For Beginners

Link prediction is like recommending friendships or connections.

**Real-world examples:**

**Social Networks:**

- Task: Friend recommendation
- Question: "Will these two users become friends?"
- How: Analyze mutual friends, shared interests, interaction patterns
- Example: "You may know..." suggestions on Facebook/LinkedIn

**E-commerce:**

- Task: Product recommendation
- Question: "Will this user purchase this product?"
- Graph: Users and products as nodes, purchases as edges
- How: Users with similar purchase history likely buy similar products

**Citation Networks:**

- Task: Predict future citations
- Question: "Will paper A cite paper B?"
- How: Analyze topic similarity, author connections, citation patterns

**Drug Discovery:**

- Task: Predict drug-target interactions
- Question: "Will this drug bind to this protein?"
- Graph: Drugs and proteins as nodes, known interactions as edges

**Key Techniques:**

- **Negative sampling**: Create non-existent edges as negative examples
- **Edge splitting**: Hide some edges during training, predict them at test time
- **Node pair scoring**: Learn to score how likely two nodes should connect

## How It Works

Link prediction aims to predict whether an edge should exist between two nodes based on:

- Node features
- Graph structure
- Edge patterns in the existing graph

## Properties

| Property | Summary |
|:-----|:--------|
| `Graph` | The graph data with edges potentially removed for training. |
| `IsDirected` | Whether the graph is directed (default: false). |
| `NegativeSamplingRatio` | Ratio of negative to positive edges for sampling. |
| `TestNegEdges` | Negative edge examples for testing. |
| `TestPosEdges` | Positive edge examples for testing. |
| `TrainNegEdges` | Negative edge examples (edges that don't exist) for training. |
| `TrainPosEdges` | Positive edge examples (edges that exist) for training. |
| `ValNegEdges` | Negative edge examples for validation. |
| `ValPosEdges` | Positive edge examples for validation. |

