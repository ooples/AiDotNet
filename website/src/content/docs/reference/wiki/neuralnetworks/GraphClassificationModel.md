---
title: "GraphClassificationModel<T>"
description: "Implements a complete neural network model for graph classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tasks.Graph`

Implements a complete neural network model for graph classification tasks.

## For Beginners

This model classifies whole graphs.

**Architecture pipeline:**

```
Step 1: Node Encoding
Input: Graph with node features
Process: Stack of GNN layers
Output: Node embeddings [num_nodes, hidden_dim]

Step 2: Graph Pooling (KEY STEP!)
Input: Node embeddings from variable-sized graph
Process: Aggregate to fixed-size representation
Output: Graph embedding [hidden_dim]

Step 3: Classification
Input: Graph embedding [hidden_dim]
Process: Fully connected layers
Output: Class probabilities [num_classes]
```

**Why pooling is crucial:**

- Graphs have variable sizes (10 nodes vs 100 nodes)
- Need fixed-size representation for classification
- Like summarizing a book (any length) into a fixed review (200 words)

**Example: Molecular toxicity prediction**
```
Molecule (graph) -> GNN layers -> Molecule embedding -> Classifier -> Toxic? (Yes/No)

Small molecule (10 atoms):
10 nodes -> GNN -> 10 embeddings -> Pool -> 1 graph embedding -> Classify

Large molecule (50 atoms):
50 nodes -> GNN -> 50 embeddings -> Pool -> 1 graph embedding -> Classify

Both produce same-sized graph embedding despite different input sizes!
```

## How It Works

Graph classification assigns labels to entire graphs based on their structure and features.
The model consists of:

1. Node-level processing (GNN layers)
2. Graph-level pooling (aggregate node embeddings)
3. Classification head (fully connected layers)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphClassificationModel` | Initializes a new instance of the `GraphClassificationModel` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets the dropout rate for regularization. |
| `EmbeddingDim` | Gets the graph embedding dimension after pooling. |
| `HiddenDim` | Gets the hidden dimension size. |
| `InputFeatures` | Gets the number of input features per node. |
| `NumClasses` | Gets the number of output classes. |
| `NumGnnLayers` | Gets the number of GNN layers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of this network type for cloning or deserialization. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data from a binary reader. |
| `EnableImplicitIdentityAdjacency` | Enables implicit self-loops-only (identity) tolerance for this model and its graph layers — the model-level analogue of GraphConvolutionalLayer's implicitIdentityWhenUnset ctor flag. |
| `EnsureDefaultAdjacencyForInput(Tensor<>)` | Auto-creates an identity adjacency matrix when none has been set, sized to the input's first dimension. |
| `EvaluateOnTask(GraphClassificationTask<>)` | Evaluates the model on test graphs. |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `GetModelMetadata` | Gets metadata about this model for serialization and identification. |
| `GetParameters` | Gets all parameters as a vector. |
| `InitializeLayers` | Initializes the layers of the neural network based on the provided architecture. |
| `PoolGraph(Tensor<>)` | Pools node embeddings into a single graph-level embedding. |
| `PredictCore(Tensor<>)` | Makes a prediction using the trained network. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data to a binary writer. |
| `SetAdjacencyMatrix(Tensor<>)` | Sets the adjacency matrix for all graph layers in the model. |
| `Train(Tensor<>,Tensor<>)` | Trains the network on a single batch of data. |
| `TrainOnTask(GraphClassificationTask<>,Int32,Double)` | Trains the model on a graph classification task. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |

