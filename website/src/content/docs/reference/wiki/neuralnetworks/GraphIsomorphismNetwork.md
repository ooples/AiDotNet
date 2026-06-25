---
title: "GraphIsomorphismNetwork<T>"
description: "Represents a Graph Isomorphism Network (GIN) for powerful graph representation learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Graph Isomorphism Network (GIN) for powerful graph representation learning.

## For Beginners

GIN is optimal for structural graph understanding.

**How it works:**

- Sum neighbor features (preserves multiset information)
- Combine with self features using learnable weighting (1 + epsilon)
- Transform through a 2-layer MLP
- Result: maximally expressive graph representation

**Example - Chemical Structure Analysis:**

- Distinguishing molecules with subtle structural differences
- GIN can tell apart molecules that simpler GNNs confuse
- Critical for drug discovery where small differences matter

**Key Features:**

- **Provably powerful**: As expressive as WL test
- **Learnable epsilon**: Optimizes self vs neighbor weighting
- **Two-layer MLP**: Provides non-linear transformation capacity
- **Sum aggregation**: Preserves structural information

**Why GIN is powerful:**

- Mean/max pooling loses information (e.g., can't distinguish {1,1,1} from {1})
- Sum aggregation preserves multiset: {1,1,1} != {1}
- MLP can approximate complex functions
- Learnable epsilon finds optimal self-weighting

**Architecture:**

1. Multiple GIN layers with sum aggregation
2. Each layer has learnable epsilon and 2-layer MLP
3. Optional graph-level readout for classification

**When to use GIN:**

- When structural differentiation is critical
- Molecular property prediction
- Chemical compound classification
- Any task where graph structure similarity matters

## How It Works

Graph Isomorphism Networks (GIN), introduced by Xu et al. (2019), are provably as powerful as
the Weisfeiler-Lehman (WL) graph isomorphism test for distinguishing graph structures. GIN uses
sum aggregation with a learnable epsilon parameter and applies a multi-layer perceptron (MLP)
for powerful feature transformation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphIsomorphismNetwork` | Initializes a new instance of the `GraphIsomorphismNetwork` class with specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InitialEpsilon` | Gets the initial epsilon value for GIN layers. |
| `IsLoRAEnabled` | Gets whether LoRA fine-tuning is currently enabled. |
| `LearnEpsilon` | Gets whether epsilon is learnable in GIN layers. |
| `LoRARank` | Gets the LoRA rank when LoRA is enabled. |
| `MlpHiddenDim` | Gets the hidden dimension size for MLP in each layer. |
| `NumLayers` | Gets the number of GIN layers in the network. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGraphLossGradient(Tensor<>,Tensor<>)` | Computes the gradient of the cross-entropy loss for graph classification. |
| `ComputeLossGradient(Tensor<>,Tensor<>,Boolean[])` | Computes the gradient of the cross-entropy loss for node classification. |
| `CreateNewInstance` | Creates a new instance of this network type. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data from a binary reader. |
| `DisableLoRA` | Disables LoRA fine-tuning and restores original layers. |
| `DistributeGradient(Tensor<>,Int32)` | Distributes gradient from graph-level back to nodes. |
| `EnableLoRAFineTuning(Int32,Double,Boolean)` | Enables LoRA fine-tuning for parameter-efficient training. |
| `Evaluate(Tensor<>,Tensor<>,Tensor<>,Boolean[])` | Evaluates the model on test data and returns accuracy for node classification. |
| `EvaluateGraphs(List<Tensor<>>,List<Tensor<>>,Tensor<>)` | Evaluates the model on test graphs and returns accuracy for graph classification. |
| `Forward(Tensor<>,Tensor<>)` | Performs a forward pass through the network with node features and adjacency matrix. |
| `ForwardForTraining(Tensor<>)` |  |
| `GetGraphRepresentation(Tensor<>,Tensor<>)` | Gets graph-level representations using sum, mean, and max pooling combined. |
| `GetLoRAParameterCount` | Gets the number of trainable LoRA parameters. |
| `GetModelMetadata` | Gets metadata about this model. |
| `GetNamedLayerActivations(Tensor<>)` | Gets the intermediate activations from each layer, ensuring adjacency is set for graph layers. |
| `GetOptions` |  |
| `GetParameterCount` | Gets the total number of trainable parameters in the network. |
| `GetParameters` | Gets all parameters as a vector. |
| `InitializeLayers` | Initializes the layers of the neural network based on the provided architecture. |
| `MergeLoRAWeights` | Merges LoRA weights into the base layers and disables LoRA mode. |
| `PredictCore(Tensor<>)` | Makes a prediction using the trained network. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data to a binary writer. |
| `SetAdjacencyMatrix(Tensor<>)` | Sets the adjacency matrix for graph operations. |
| `SumReadout(Tensor<>)` | Sum readout for graph-level representation. |
| `Train(Tensor<>,Tensor<>)` | Trains the network on a single batch of data via tape-based autodiff. |
| `TrainOnGraph(Tensor<>,Tensor<>,Tensor<>,Boolean[],Int32,Double)` | Trains the GIN network on a single graph with node classification. |
| `TrainOnGraphs(List<Tensor<>>,List<Tensor<>>,Tensor<>,Int32,Double)` | Trains the GIN network on multiple graphs for graph classification. |
| `TrainStepWithAdjacency(Tensor<>,Tensor<>,Tensor<>)` | Shared tape-backed training step that pins the supplied adjacency matrix on the network and routes through `Tensor{`. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cachedAdjacencyMatrix` | Cached adjacency matrix for forward/backward passes. |
| `_lossFunction` | The loss function used to calculate the error between predicted and expected outputs. |
| `_optimizer` | The optimization algorithm used to update the network's parameters during training. |

