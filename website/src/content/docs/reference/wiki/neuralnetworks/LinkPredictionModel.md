---
title: "LinkPredictionModel<T>"
description: "Implements a complete neural network model for link prediction tasks on graphs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tasks.Graph`

Implements a complete neural network model for link prediction tasks on graphs.

## For Beginners

This model predicts connections between nodes.

**How it works:**

1. **Encode**: Learn embeddings for all nodes using GNN layers

```
Input: Node features + Graph structure
Process: Stack of graph conv layers
Output: Node embeddings [num_nodes, embedding_dim]
```

2. **Decode**: Score node pairs to predict edges

```
Input: Node pair (i, j)
Compute: score = f(embedding[i], embedding[j])
Common functions:

- Dot product: z_i * z_j
- Concatenation + MLP: MLP([z_i || z_j])
- Distance-based: -||z_i - z_j||^2

```

3. **Train**: Learn to score existing edges high, non-existing edges low

**Example:**
```
Friend recommendation:

- Encode users as embeddings using friend network
- For user pair (Alice, Bob):
* Compute score from their embeddings
* High score -> Likely to be friends
* Low score -> Unlikely to be friends

```

## How It Works

Link prediction predicts whether edges should exist between node pairs using:

- Node features
- Graph structure
- Learned node embeddings

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinkPredictionModel` | Initializes a new instance of the `LinkPredictionModel` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets the dropout rate for regularization. |
| `EmbeddingDim` | Gets the embedding dimension. |
| `HiddenDim` | Gets the hidden dimension size. |
| `InputFeatures` | Gets the number of input features per node. |
| `NumLayers` | Gets the number of GNN layers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of this network type for cloning or deserialization. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data from a binary reader. |
| `EnableImplicitIdentityAdjacency` | Enables implicit self-loops-only (identity) tolerance for this model and its graph layers — the model-level analogue of GraphConvolutionalLayer's implicitIdentityWhenUnset ctor flag. |
| `EnsureDefaultAdjacencyForInput(Tensor<>)` | Default-of-last-resort adjacency: when no explicit matrix was set, fall back to the IDENTITY graph so scaffold invariants stay well-defined — per Kipf & Welling 2017 §2, with `A = I` the GCN encoder degenerates to a per-node dense transform… |
| `EvaluateOnTask(LinkPredictionTask<>)` | Evaluates the model on test edges. |
| `Forward(Tensor<>)` | Performs a forward pass through the encoder network. |
| `GetModelMetadata` | Gets metadata about this model for serialization and identification. |
| `GetParameters` | Gets all parameters as a vector. |
| `InitializeLayers` | Initializes the layers of the neural network based on the provided architecture. |
| `PredictCore(Tensor<>)` | Makes a prediction using the trained network. |
| `PredictEdges(Tensor<>)` | Computes edge scores for given node pairs. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data to a binary writer. |
| `SetAdjacencyMatrix(Tensor<>)` | Sets the adjacency matrix for all graph layers in the model. |
| `Train(Tensor<>,Tensor<>)` | Trains the network on a single batch of data. |
| `TrainOnTask(LinkPredictionTask<>,Int32,Double)` | Trains the model on a link prediction task. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |

