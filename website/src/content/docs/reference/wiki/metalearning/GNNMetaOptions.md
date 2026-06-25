---
title: "GNNMetaOptions<T, TInput, TOutput>"
description: "Configuration options for the Graph Neural Network Meta-learning algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for the Graph Neural Network Meta-learning algorithm.

## For Beginners

GNN Meta-learning treats the learning process as a graph problem:

1. Each task or example becomes a node in a graph
2. Relationships between tasks are edges (e.g., task similarity)
3. A graph neural network learns to propagate useful information
4. This allows the model to leverage task relationships for better adaptation

Imagine a social network where users are tasks and friendships are similarities.
By looking at what similar tasks learned, new tasks can adapt faster.

## How It Works

GNN-based meta-learning models tasks and examples as nodes in a graph,
with edges representing relationships between them. The graph neural network
learns to propagate information across the task structure to improve learning.

**Key Components:**

- **Node Embeddings:** Represent tasks/examples as vectors
- **Message Passing:** Share information between connected nodes
- **Graph Aggregation:** Combine node information for predictions

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GNNMetaOptions(IFullModel<,,>)` | Initializes a new instance of the GNNMetaOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of gradient steps to take during inner loop adaptation. |
| `AggregationType` | Gets or sets the aggregation method for graph-level representations. |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `DropoutRate` | Gets or sets the dropout rate for GNN layers. |
| `EdgeFeatureDimension` | Gets or sets the dimension of edge features in the task graph. |
| `EdgeThreshold` | Gets or sets the threshold for creating edges between tasks. |
| `EnableCheckpointing` | Gets or sets whether to save checkpoints during training. |
| `EvaluationFrequency` | Gets or sets how often to evaluate during meta-training. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `GNNHiddenDimension` | Gets or sets the hidden dimension for the GNN layers. |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (task adaptation). |
| `InnerOptimizer` | Gets or sets the optimizer for inner-loop adaptation. |
| `LearnEdgeWeights` | Gets or sets whether to learn edge weights during training. |
| `LossFunction` | Gets or sets the loss function for training. |
| `MaxNeighbors` | Gets or sets the maximum number of neighbors per node in sparse graph. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for meta-parameter updates (outer loop). |
| `NodeEmbeddingDimension` | Gets or sets the dimension of node embeddings in the task graph. |
| `NumAttentionHeads` | Gets or sets the number of attention heads for graph attention. |
| `NumMessagePassingLayers` | Gets or sets the number of message passing layers in the GNN. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations to perform. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (meta-update). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `SimilarityMetric` | Gets or sets how task similarity is computed for building the task graph. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation. |
| `UseFullyConnectedGraph` | Gets or sets whether to use a fully connected task graph. |
| `UseLayerNorm` | Gets or sets whether to use layer normalization in GNN layers. |
| `UseResidualConnections` | Gets or sets whether to use residual connections in GNN layers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the GNN Meta options. |
| `IsValid` | Validates that all GNN Meta configuration options are properly set. |

