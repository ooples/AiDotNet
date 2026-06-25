---
title: "GraphSAGENetwork<T>"
description: "Represents a GraphSAGE (Graph Sample and Aggregate) Network for inductive learning on graphs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a GraphSAGE (Graph Sample and Aggregate) Network for inductive learning on graphs.

## For Beginners

GraphSAGE learns how to combine neighbor information.

**How it works:**

- For each node, sample its neighbors
- Aggregate neighbor features using a learnable function
- Combine with node's own features
- Result: new representation that captures local structure

**Example - Social Network Recommendations:**

- New user joins the platform (unseen during training)
- GraphSAGE can still make recommendations by:
1. Looking at the new user's connections
2. Aggregating features from those connections
3. Generating a representation for the new user

**Key Features:**

- **Inductive**: Can generalize to new, unseen nodes
- **Scalable**: Uses sampling, not full neighborhoods
- **Flexible aggregators**: Mean, MaxPool, or Sum
- **L2 normalization**: Optional for stable training

**Aggregator Types:**

- **Mean**: Average of neighbor features (most common)
- **MaxPool**: Element-wise max (captures salient features)
- **Sum**: Sum of neighbor features (preserves structure)

**Architecture:**

1. Multiple GraphSAGE layers with different aggregators
2. Optional L2 normalization between layers
3. Final classification or regression head

**When to use GraphSAGE:**

- When new nodes appear frequently (evolving graphs)
- When you need to generalize to new graphs
- For large-scale graphs where full-batch training is infeasible
- Social networks, recommendation systems, dynamic graphs

## How It Works

GraphSAGE, introduced by Hamilton et al. (2017), is designed for inductive learning on graphs.
Unlike transductive methods that require all nodes during training, GraphSAGE learns aggregator
functions that can generalize to completely unseen nodes and even new graphs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphSAGENetwork` | Initializes a new instance of the `GraphSAGENetwork` class with specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AggregatorType` | Gets the aggregator type used in GraphSAGE layers. |
| `DropoutRate` | Gets the dropout rate applied during training. |
| `HiddenDim` | Gets the hidden dimension size for each layer. |
| `IsLoRAEnabled` | Gets whether LoRA fine-tuning is currently enabled. |
| `LoRARank` | Gets the LoRA rank when LoRA is enabled. |
| `Normalize` | Gets whether L2 normalization is applied after each layer. |
| `NumLayers` | Gets the number of GraphSAGE layers in the network. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLossGradient(Tensor<>,Tensor<>,Boolean[])` | Computes the gradient of the cross-entropy loss. |
| `CreateNewInstance` | Creates a new instance of this network type. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data from a binary reader. |
| `DisableLoRA` | Disables LoRA fine-tuning and restores original layers. |
| `EnableLoRAFineTuning(Int32,Double,Boolean)` | Enables LoRA fine-tuning for parameter-efficient training. |
| `Evaluate(Tensor<>,Tensor<>,Tensor<>,Boolean[])` | Evaluates the model on test data and returns accuracy. |
| `Forward(Tensor<>,Tensor<>)` | Performs a forward pass through the network with node features and adjacency matrix. |
| `GetLoRAParameterCount` | Gets the number of trainable LoRA parameters. |
| `GetModelMetadata` | Gets metadata about this model. |
| `GetNamedLayerActivations(Tensor<>)` | Gets the intermediate activations from each layer, ensuring adjacency is set for graph layers. |
| `GetNodeEmbeddings(Tensor<>,Tensor<>)` | Generates node embeddings using the trained network. |
| `GetOptions` |  |
| `GetParameterCount` | Gets the total number of trainable parameters in the network. |
| `GetParameters` | Gets all parameters as a vector. |
| `InitializeLayers` | Initializes the layers of the neural network based on the provided architecture. |
| `MergeLoRAWeights` | Merges LoRA weights into the base layers and disables LoRA mode. |
| `PredictCore(Tensor<>)` | Makes a prediction using the trained network. |
| `SampleSubgraph(Tensor<>,Tensor<>,Tensor<>,Int32[],Int32,Random)` | Samples a subgraph around target nodes for mini-batch training. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data to a binary writer. |
| `SetAdjacencyMatrix(Tensor<>)` | Sets the adjacency matrix for graph operations. |
| `Train(Tensor<>,Tensor<>)` | Trains the network on a single batch of data. |
| `TrainMiniBatch(Tensor<>,Tensor<>,Tensor<>,Int32[],Int32,Int32,Double,Int32)` | Performs mini-batch training using neighbor sampling for scalability. |
| `TrainOnGraph(Tensor<>,Tensor<>,Tensor<>,Boolean[],Int32,Double)` | Trains the GraphSAGE network on graph-structured data. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cachedAdjacencyMatrix` | Cached adjacency matrix for forward/backward passes. |
| `_lossFunction` | The loss function used to calculate the error between predicted and expected outputs. |
| `_optimizer` | The optimization algorithm used to update the network's parameters during training. |

