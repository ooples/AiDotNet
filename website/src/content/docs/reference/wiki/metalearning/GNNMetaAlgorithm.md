---
title: "GNNMetaAlgorithm<T, TInput, TOutput>"
description: "Implementation of Graph Neural Network-based Meta-learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Graph Neural Network-based Meta-learning.

## For Beginners

GNN Meta-learning is like studying with a study group:

## How It Works

GNN-based meta-learning models tasks and examples as nodes in a graph,
with edges representing relationships between them. The graph neural network
learns to propagate information across the task structure to improve learning.

**Key Innovation:** Instead of treating tasks independently, GNN Meta-learning:

1. Builds a graph where nodes represent tasks or examples
2. Edges connect similar or related tasks
3. Message passing propagates useful information between tasks
4. The aggregated graph information guides adaptation

- MAML: Each student learns alone but starts with good study habits
- GNN Meta: Students share notes and help each other learn faster

When learning a new subject (task), you can benefit from what others
who studied similar subjects (similar tasks) have learned. The graph
network learns which tasks are helpful for each other.

**Architecture:**

- **Node Embeddings:** Each task gets a vector representation
- **Edge Weights:** Learned weights showing task relationships
- **Message Passing:** Information flows between related tasks
- **Graph Aggregation:** Combines all node information for prediction

**Algorithm:**

Reference: Garcia, V., & Bruna, J. (2018). Few-shot learning with graph neural networks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GNNMetaAlgorithm(GNNMetaOptions<,,>)` | Initializes a new instance of the GNNMetaAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` | Gets the algorithm type identifier for this meta-learner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts the meta-learned model to a new task using graph-informed adaptation. |
| `AdaptWithGraphContext(IFullModel<,,>,IMetaLearningTask<,,>,Vector<>)` | Adapts model using graph context. |
| `AggregateGraphInformation(List<Vector<>>,Matrix<>)` | Aggregates graph information to provide context for each task. |
| `AggregateNeighborMessages(Int32,List<Vector<>>,Matrix<>)` | Aggregates messages from neighboring nodes. |
| `BuildTaskGraph(TaskBatch<,,>)` | Builds a task graph from the current batch of tasks. |
| `ComputeAttentionContext(List<Vector<>>,Vector<>)` | Computes attention-weighted context from node embeddings. |
| `ComputeBatchLoss(TaskBatch<,,>)` | Computes loss on a batch for gradient estimation. |
| `ComputeLearnedSimilarity(Vector<>,Vector<>)` | Computes learned similarity using edge weights. |
| `ComputeMeanContext(List<Vector<>>,Int32,Matrix<>)` | Computes mean context from neighbor embeddings using the adjacency matrix. |
| `ComputeMeanPartial(Vector<>,Int32)` | Computes mean of first n elements. |
| `ComputeTaskEmbedding(IMetaLearningTask<,,>)` | Computes an embedding for a task from its support set. |
| `ComputeTaskSimilarity(Vector<>,Vector<>)` | Computes similarity between two task embeddings. |
| `FindMostSimilarContext(Vector<>)` | Finds the most similar cached context for a new task. |
| `InitializeWeights(Int32)` | Initializes weights using Xavier initialization. |
| `InnerLoopAdaptation(IFullModel<,,>,IMetaLearningTask<,,>)` | Performs inner loop adaptation on a task. |
| `MessagePassing(List<Vector<>>,Matrix<>)` | Performs message passing on the task graph. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step using GNN-based task relationship modeling. |
| `ModulateGradientsWithContext(Vector<>,Vector<>)` | Modulates gradients using graph context. |
| `TransformEmbedding(Vector<>,Int32)` | Transforms embedding using message passing weights with a two-layer projection. |
| `UpdateGNNWeights(TaskBatch<,,>,)` | Updates GNN weights using finite differences with scaled gradient estimation. |

