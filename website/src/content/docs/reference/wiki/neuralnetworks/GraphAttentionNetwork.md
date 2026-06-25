---
title: "GraphAttentionNetwork<T>"
description: "Represents a Graph Attention Network (GAT) that uses attention mechanisms to process graph-structured data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Graph Attention Network (GAT) that uses attention mechanisms to process graph-structured data.

## For Beginners

GAT is like having a smart filter for your social network.

**How it works:**

- Each node looks at its neighbors and decides which ones are most important
- Important neighbors get more "attention" (higher weights)
- Less relevant neighbors get less attention

**Example - Movie Recommendations:**

- You're a node connected to movies you've watched
- Some movies better represent your taste than others
- GAT learns to pay more attention to movies that define your preferences
- Result: Better recommendations by focusing on what matters most

**Key Features:**

- **Multi-head attention**: Multiple attention "perspectives" for robustness
- **Dynamic weights**: Attention weights are learned, not fixed
- **Dropout support**: Prevents overfitting during training
- **Configurable heads**: Adjust number of attention heads for your task

**Architecture:**
The standard GAT architecture consists of:

1. Multiple GAT layers with attention mechanisms
2. Optional dropout between layers
3. Final classification or regression head

**When to use GAT:**

- When some neighbors are more informative than others
- When you need interpretable importance scores
- For heterogeneous graphs where relationships vary in importance
- Citation networks, social networks, knowledge graphs

## How It Works

Graph Attention Networks introduce attention mechanisms to graph neural networks, allowing the model
to learn which neighbors are most important for each node. Unlike GCN which treats all neighbors equally,
GAT learns attention weights that determine how much each neighbor contributes to a node's representation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphAttentionNetwork` | Initializes a new instance of the `GraphAttentionNetwork` class with specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets the dropout rate applied to attention coefficients during training. |
| `HiddenDim` | Gets the hidden dimension size for each layer. |
| `IsLoRAEnabled` | Gets whether LoRA fine-tuning is currently enabled. |
| `LoRARank` | Gets the LoRA rank when LoRA is enabled. |
| `NumHeads` | Gets the number of attention heads used in each GAT layer. |
| `NumLayers` | Gets the number of GAT layers in the network. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLossGradient(Tensor<>,Tensor<>,Boolean[])` | Computes the gradient of the cross-entropy loss. |
| `CreateNewInstance` | Creates a new instance of this network type for cloning or deserialization. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data from a binary reader. |
| `DisableLoRA` | Disables LoRA fine-tuning and restores original layers. |
| `EnableLoRAFineTuning(Int32,Double,Boolean)` | Enables LoRA (Low-Rank Adaptation) fine-tuning for parameter-efficient training. |
| `Evaluate(Tensor<>,Tensor<>,Tensor<>,Boolean[])` | Evaluates the model on test data and returns accuracy. |
| `Forward(Tensor<>,Tensor<>)` | Performs a forward pass through the network with node features and adjacency matrix. |
| `ForwardForTraining(Tensor<>)` |  |
| `GetAttentionWeights` | Gets attention weights from all GAT layers for interpretability. |
| `GetLoRAParameterCount` | Gets the number of trainable LoRA parameters when LoRA is enabled. |
| `GetLoRATrainablePercentage` | Gets the percentage of parameters that are trainable when using LoRA. |
| `GetModelMetadata` | Gets metadata about this model for serialization and identification. |
| `GetNamedLayerActivations(Tensor<>)` | Gets the intermediate activations from each layer, ensuring adjacency is set for graph layers. |
| `GetOptions` |  |
| `GetParameterCount` | Gets the total number of trainable parameters in the network. |
| `GetParameters` | Gets all parameters as a vector. |
| `InitializeLayers` | Initializes the layers of the neural network based on the provided architecture. |
| `MergeLoRAWeights` | Merges LoRA weights into the base layers and disables LoRA mode. |
| `PredictCore(Tensor<>)` | Makes a prediction using the trained network. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data to a binary writer. |
| `SetAdjacencyMatrix(Tensor<>)` | Sets the adjacency matrix for graph operations. |
| `Train(Tensor<>,Tensor<>)` | Trains the network on a single batch of data. |
| `TrainOnGraph(Tensor<>,Tensor<>,Tensor<>,Boolean[],Int32,Double)` | Trains the GAT network on graph-structured data. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cachedAdjacencyMatrix` | Cached adjacency matrix for forward/backward passes. |
| `_lossFunction` | The loss function used to calculate the error between predicted and expected outputs. |
| `_optimizer` | The optimization algorithm used to update the network's parameters during training. |

