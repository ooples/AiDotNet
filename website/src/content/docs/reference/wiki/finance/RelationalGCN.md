---
title: "RelationalGCN<T>"
description: "RelationalGCN (Relational Graph Convolutional Network) for multi-relational graph learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Graph`

RelationalGCN (Relational Graph Convolutional Network) for multi-relational graph learning.

## For Beginners

RelationalGCN is designed for knowledge graphs:

**The Key Insight:**
Standard GCN treats all edges equally, but in knowledge graphs and financial networks,
different types of relationships matter differently. A "supplies-to" relationship is
fundamentally different from a "competes-with" relationship. R-GCN learns separate
transformations for each relation type.

**What Problems Does RelationalGCN Solve?**

- Entity classification in knowledge graphs (company type, sector classification)
- Link prediction in multi-relational networks (predicting missing relationships)
- Financial network analysis with multiple relationship types
- Supply chain modeling with different connection types

**How RelationalGCN Works:**

1. **Relation-Specific Weights:** Learns different weights W_r for each relation type r
2. **Basis Decomposition:** W_r = sum_b (a_rb * B_b) shares parameters via learned bases
3. **Block Decomposition:** Alternative using block-diagonal weight matrices
4. **Self-Connections:** Special weight W_0 for a node's own features

**RelationalGCN Architecture:**

- Message Passing: h_i^(l+1) = sigma(sum_r sum_j A_r[i,j] * h_j^(l) * W_r + h_i^(l) * W_0)
- Basis Decomposition: W_r = sum_b (a_rb * B_b) with shared bases B
- Block Decomposition: W_r = diag(W_r1, ..., W_rB) with block-diagonal structure

**Key Benefits:**

- Handles heterogeneous graphs with multiple edge types
- Parameter efficient through basis or block decomposition
- Captures relation-specific patterns in the data
- Effective for both entity classification and link prediction

## How It Works

RelationalGCN extends Graph Convolutional Networks to handle multi-relational data
where different types of edges (relations) exist between nodes, making it ideal for
knowledge graphs and heterogeneous networks.

**Reference:** Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks", ESWC 2018.
https://arxiv.org/abs/1703.06103

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RelationalGCN(NeuralNetworkArchitecture<>,RelationalGCNOptions<>,Double[0:,0:][],IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the RelationalGCN model in native mode for training. |
| `RelationalGCN(NeuralNetworkArchitecture<>,String,RelationalGCNOptions<>,Double[0:,0:][],IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the RelationalGCN model in ONNX mode for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ForecastHorizon` | Gets the forecast horizon. |
| `IsChannelIndependent` |  |
| `NumBases` | Gets the number of basis matrices for basis decomposition. |
| `NumFeatures` |  |
| `NumNodes` | Gets the number of nodes (entities) in the knowledge graph. |
| `NumRelations` | Gets the number of relation types. |
| `NumSamples` | Gets the number of samples for uncertainty estimation. |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` | Gets whether the model supports training (native mode only). |
| `UseBasisDecomposition` | Gets whether basis decomposition is used. |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateAcrossRelations(Double[0:,0:])` | Aggregates messages across all relation types. |
| `ApplyInstanceNormalization(Tensor<>)` | Applies instance normalization (identity for RelationalGCN). |
| `ApplyRelationalConvolution(Double[0:,0:],Int32)` | Applies relational graph convolution for one relation type. |
| `AutoregressiveForecast(Tensor<>,Int32)` | Performs autoregressive forecasting for extended horizons. |
| `ComputePredictionIntervals(List<Tensor<>>,Double)` | Computes prediction intervals from samples. |
| `ComputeQuantiles(List<Tensor<>>,Double[])` | Computes quantiles from samples. |
| `ConcatenatePredictions(List<Tensor<>>)` | Concatenates multiple predictions into one tensor. |
| `CreateDefaultRelationAdjacencies` | Creates default relation adjacencies when none provided. |
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes RelationalGCN-specific data. |
| `Dispose(Boolean)` | Disposes managed resources. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates forecast quality against actual values. |
| `ExtractLayerReferences` | Extracts references to specific layer types for direct access. |
| `FlattenInput(Tensor<>)` | Flattens input tensor for layer processing. |
| `Forecast(Tensor<>,Double[])` | Generates forecasts for all nodes. |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `ForecastWithIntervals(Tensor<>,Double)` | Generates forecasts with prediction intervals. |
| `Forward(Tensor<>)` | Performs the forward pass through all layers. |
| `GenerateSamples(Tensor<>,Int32)` | Generates MC Dropout samples for uncertainty estimation. |
| `GetFinancialMetrics` | Gets financial-specific metrics. |
| `GetModelMetadata` | Gets metadata about the RelationalGCN model. |
| `GetOptions` |  |
| `GetRelationWeight(Int32)` | Computes relation-specific weight matrix using basis decomposition. |
| `InitializeBasisDecomposition` | Initializes the basis decomposition matrices. |
| `InitializeLayers` | Initializes the neural network layers for RelationalGCN. |
| `NormalizeAdjacency(Double[0:,0:])` | Normalizes an adjacency matrix by row sum. |
| `PredictCore(Tensor<>)` | Performs forward prediction on the input tensor. |
| `ReshapeOutput(Tensor<>)` | Reshapes output to forecast format. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes RelationalGCN-specific data. |
| `ShiftInputWindow(Tensor<>,Tensor<>)` | Shifts the input window by appending a prediction. |
| `Train(Tensor<>,Tensor<>)` | Trains the RelationalGCN model on a batch of input-target pairs. |
| `UpdateBasisDecompositionFromGradient(Vector<>)` | Updates basis decomposition parameters based on loss gradient. |
| `UpdateParameters(Vector<>)` | Updates parameters using the provided gradients. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers provided by the user. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_basisMatrices` | Learned basis matrices for basis decomposition. |
| `_relationAdjacencies` | Adjacency matrices for each relation type. |
| `_relationCoefficients` | Learned coefficients for combining basis matrices per relation. |

