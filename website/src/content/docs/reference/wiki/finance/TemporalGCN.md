---
title: "TemporalGCN<T>"
description: "TemporalGCN (Temporal Graph Convolutional Network) for time series forecasting on graph-structured data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Graph`

TemporalGCN (Temporal Graph Convolutional Network) for time series forecasting on graph-structured data.

## For Beginners

TemporalGCN learns two types of patterns simultaneously:

**The Key Insight:**
Many time series exist on networks: traffic sensors on roads, users in social networks,
weather stations across regions. TemporalGCN captures both HOW these entities are connected
(spatial) and HOW they change over time (temporal).

**What Problems Does TemporalGCN Solve?**

- Traffic flow prediction (sensors connected by roads)
- Social network activity forecasting (users connected by friendships)
- Epidemiological prediction (regions connected geographically)
- Financial network analysis (assets connected by correlations)

**How TemporalGCN Works:**

1. **Graph Convolution:** Aggregate information from neighboring nodes
2. **Temporal Recurrence:** Process time sequences with GRU cells
3. **Interleaved Processing:** Alternate between spatial and temporal layers
4. **Prediction:** Output forecasts for all nodes simultaneously

**TemporalGCN Architecture:**

- GCN Layers: Chebyshev spectral graph convolution for neighbor aggregation
- GRU Layers: Gated recurrent units for temporal sequence modeling
- Batch Normalization: Stabilizes training across layers
- Residual Connections: Helps gradients flow through deep networks

**Key Benefits:**

- Jointly learns spatial and temporal dependencies
- Handles variable graph structures (nodes can have different numbers of neighbors)
- Computationally efficient with Chebyshev polynomial approximation
- Works with both static and dynamic graphs

## How It Works

TemporalGCN combines graph convolutional networks with recurrent neural networks (GRU/LSTM)
to capture both spatial dependencies and temporal dynamics in graph-structured time series.

**Reference:** Zhao et al., "T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction", 2019.
https://arxiv.org/abs/1811.05320

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TemporalGCN(NeuralNetworkArchitecture<>,String,TemporalGCNOptions<>,Double[0:,0:],IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the TemporalGCN model in ONNX mode for inference. |
| `TemporalGCN(NeuralNetworkArchitecture<>,TemporalGCNOptions<>,Double[0:,0:],IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the TemporalGCN model in native mode for training. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ChebyshevOrder` | Gets the Chebyshev polynomial order for graph convolution. |
| `ForecastHorizon` | Gets the forecast horizon. |
| `IsChannelIndependent` |  |
| `NumFeatures` |  |
| `NumNodes` | Gets the number of nodes in the graph. |
| `NumSamples` | Gets the number of samples for uncertainty estimation. |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` | Gets whether the model supports training (native mode only). |
| `TemporalCellType` | Gets the type of temporal recurrent cell used. |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddSmallPerturbation(Tensor<>)` | Adds small random perturbation to input for sample diversity. |
| `ApplyChebyshevConvolution(Tensor<>)` | Applies Chebyshev spectral graph convolution. |
| `ApplyChebyshevConvolutionTape(Tensor<>)` | Tape-aware Chebyshev (K=2) graph filter: `y = 0.5 * x + 0.5 * L x`. |
| `ApplyInstanceNormalization(Tensor<>)` | Applies instance normalization (identity for TemporalGCN). |
| `AutoregressiveForecast(Tensor<>,Int32)` | Performs autoregressive forecasting for extended horizons. |
| `ComputeNormalizedLaplacian(Double[0:,0:])` | Computes the normalized graph Laplacian for Chebyshev convolution. |
| `ComputePredictionIntervals(List<Tensor<>>,Double)` | Computes prediction intervals from samples. |
| `ComputeQuantiles(List<Tensor<>>,Double[])` | Computes quantiles from samples. |
| `ConcatenatePredictions(List<Tensor<>>)` | Concatenates multiple predictions. |
| `CreateDefaultAdjacencyMatrix(Int32)` | Creates a default adjacency matrix for the graph. |
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes TemporalGCN-specific data. |
| `Dispose(Boolean)` | Disposes of resources used by the model. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates forecast quality against actual values. |
| `ExtractLayerReferences` | Extracts references to key layers for efficient access. |
| `FlattenInput(Tensor<>)` | Flattens input tensor for dense layer processing. |
| `Forecast(Tensor<>,Double[])` | Generates forecasts for all nodes in the graph. |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `ForecastWithIntervals(Tensor<>,Double)` | Generates forecasts with prediction intervals. |
| `Forward(Tensor<>)` | Performs the forward pass through all layers. |
| `ForwardNativeForTraining(Tensor<>)` | Trains the TemporalGCN model on a batch of input-target pairs. |
| `GenerateSamples(Tensor<>,Int32)` | Generates multiple forecast samples using MC Dropout. |
| `GetFinancialMetrics` | Gets financial-specific metrics. |
| `GetModelMetadata` | Gets metadata about the TemporalGCN model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes all layers for the TemporalGCN model. |
| `PredictCore(Tensor<>)` | Performs forward prediction on the input tensor. |
| `ReshapeToNodes(Tensor<>)` | Reshapes the input to [numNodes, featuresPerNode] (numNodes as the batch dimension) via the tape-aware Engine.Reshape so the per-node MLP layers apply SHARED weights across nodes and gradients flow back to the input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes TemporalGCN-specific data. |
| `ShiftInputWindow(Tensor<>,Tensor<>)` | Shifts input window by appending new prediction. |
| `UpdateParameters(Vector<>)` | Updates parameters using the provided gradients. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers provided by the user. |

