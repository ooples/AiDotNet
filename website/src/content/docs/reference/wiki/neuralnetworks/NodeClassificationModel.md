---
title: "NodeClassificationModel<T>"
description: "Implements a complete neural network model for node classification tasks on graphs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tasks.Graph`

Implements a complete neural network model for node classification tasks on graphs.

## For Beginners

This model classifies nodes in a graph.

**How it works:**

1. **Input**: Graph with node features and structure
2. **Processing**: Stack of graph convolutional layers
- Each layer aggregates information from neighbors
- Features become more "context-aware" at each layer
- After k layers, each node knows about its k-hop neighborhood
3. **Output**: Class predictions for each node

**Example architecture:**
```
Input: [num_nodes, input_features]
|
GCN Layer 1: [num_nodes, hidden_dim]
|
Activation (ReLU)
|
Dropout
|
GCN Layer 2: [num_nodes, num_classes]
|
Softmax: [num_nodes, num_classes] (probabilities)
```

**Training:**

- Use labeled nodes for computing loss
- Unlabeled nodes still participate in message passing
- Graph structure helps propagate label information

## How It Works

Node classification predicts labels for individual nodes in a graph using:

- Node features
- Graph structure (adjacency information)
- Semi-supervised learning (only some nodes have labels)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NodeClassificationModel` | Initializes a new instance of the `NodeClassificationModel` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets the dropout rate for regularization. |
| `HiddenDim` | Gets the hidden dimension size. |
| `InputFeatures` | Gets the number of input features per node. |
| `NumClasses` | Gets the number of output classes. |
| `NumLayers` | Gets the number of graph layers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of this network type for cloning or deserialization. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data from a binary reader. |
| `EnableImplicitIdentityAdjacency` | Enables implicit self-loops-only (identity) tolerance for this model and its graph layers — the model-level analogue of GraphConvolutionalLayer's implicitIdentityWhenUnset ctor flag. |
| `EnsureDefaultAdjacencyForInput(Tensor<>)` | Auto-creates an identity adjacency matrix when none has been set — see `GraphClassificationModel` for rationale (test- scaffold convenience, paper-faithful degenerate case where the GCN degrades to a per-node dense transform under Kipf & We… |
| `EvaluateOnTask(NodeClassificationTask<>)` | Evaluates the model on test nodes. |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `GetModelMetadata` | Gets metadata about this model for serialization and identification. |
| `GetParameters` | Gets all parameters as a vector. |
| `InitializeLayers` | Initializes the layers of the neural network based on the provided architecture. |
| `PredictCore(Tensor<>)` | Makes a prediction using the trained network. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data to a binary writer. |
| `SetAdjacencyMatrix(Tensor<>)` | Sets the adjacency matrix for all graph layers in the model. |
| `Train(Tensor<>,Tensor<>)` | Trains the network on a single batch of data. |
| `TrainOnTask(NodeClassificationTask<>,Int32,Double)` | Trains the model on a node classification task. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |

