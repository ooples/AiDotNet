---
title: "STGNN<T>"
description: "STGNN (Spatio-Temporal Graph Neural Network) for forecasting on graph-structured time series data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Graph`

STGNN (Spatio-Temporal Graph Neural Network) for forecasting on graph-structured time series data.

## For Beginners

STGNN is designed for data where locations/entities are connected:

**The Key Insight:**
Many real-world time series are not independent - they're connected in space or by relationships.
Traffic at one intersection affects nearby intersections; one stock's movement affects related stocks.
STGNN models both these spatial connections and temporal patterns simultaneously.

**What Problems Does STGNN Solve?**

- Traffic forecasting (sensors connected by roads)
- Financial network prediction (assets connected by correlations)
- Weather forecasting (stations connected geographically)
- Social network dynamics (users connected by relationships)

**How STGNN Works:**

1. **Graph Representation:** Encode spatial relationships as an adjacency matrix
2. **Spatial Aggregation:** Each node gathers information from neighbors via graph convolution
3. **Temporal Modeling:** Capture time patterns using temporal convolutions
4. **ST Fusion:** Alternate spatial and temporal processing for joint modeling
5. **Prediction:** Output forecasts for all nodes in the network

**STGNN Architecture:**

- ST-Conv Blocks: Sandwich structure (Temporal-Spatial-Temporal) for deep ST learning
- Graph Convolution: Chebyshev spectral convolution for efficient neighbor aggregation
- Gated Units: Control information flow between spatial and temporal paths
- Residual Connections: Enable training of deep networks

**Key Benefits:**

- Captures complex spatio-temporal dependencies
- Scales to large graphs (hundreds of nodes)
- Handles both directed and undirected graphs
- Provides multi-step ahead forecasts for all nodes

## How It Works

STGNN combines graph neural networks for spatial dependencies with temporal convolutions
for time series patterns, enabling forecasting on interconnected entities.

**Reference:** Yu et al., "Spatio-Temporal Graph Convolutional Networks", IJCAI 2018.
https://arxiv.org/abs/1709.04875

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `STGNN(NeuralNetworkArchitecture<>,STGNNOptions<>,Double[0:,0:],IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the STGNN model in native mode for training. |
| `STGNN(NeuralNetworkArchitecture<>,String,STGNNOptions<>,Double[0:,0:],IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the STGNN model in ONNX mode for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ForecastHorizon` | Gets the forecast horizon. |
| `GraphConvType` | Gets the type of graph convolution used. |
| `IsChannelIndependent` |  |
| `NumFeatures` |  |
| `NumNodes` | Gets the number of nodes in the graph. |
| `NumSamples` | Gets the number of samples for uncertainty estimation. |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` | Gets whether the model supports training (native mode only). |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddSmallPerturbation(Tensor<>)` | Adds small random perturbation to input for sample diversity. |
| `ApplyGraphConvolution(Tensor<>)` | Applies graph convolution using the adjacency matrix. |
| `ApplyInstanceNormalization(Tensor<>)` | Applies instance normalization (identity for STGNN). |
| `AutoregressiveForecast(Tensor<>,Int32)` | Performs autoregressive forecasting for extended horizons. |
| `BuildConstantMatrix(Double[0:,0:])` | Builds a constant (non-trainable) `Tensor` from the adjacency matrix so it can participate in tape-aware matmuls without being treated as a learnable parameter. |
| `ComputePredictionIntervals(List<Tensor<>>,Double)` | Computes prediction intervals from samples. |
| `ComputeQuantiles(List<Tensor<>>,Double[])` | Computes quantiles from samples. |
| `ConcatenatePredictions(List<Tensor<>>)` | Concatenates multiple predictions. |
| `CreateDefaultAdjacencyMatrix(Int32)` | Creates a default adjacency matrix for the graph. |
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes STGNN-specific data. |
| `Dispose(Boolean)` | Disposes of resources. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates forecast quality against actual values. |
| `ExtractLayerReferences` | Extracts references to key layers for efficient access. |
| `FlattenInput(Tensor<>)` | Flattens input tensor for dense layer processing. |
| `Forecast(Tensor<>,Double[])` | Generates forecasts for all nodes in the graph. |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `ForecastWithIntervals(Tensor<>,Double)` | Generates forecasts with prediction intervals. |
| `Forward(Tensor<>)` | Performs the forward pass through all layers. |
| `GenerateSamples(Tensor<>,Int32)` | Generates multiple forecast samples using MC Dropout. |
| `GetFinancialMetrics` | Gets financial-specific metrics. |
| `GetModelMetadata` | Gets metadata about the STGNN model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes all layers for the STGNN model. |
| `PredictCore(Tensor<>)` | Performs forward prediction on the input tensor. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes STGNN-specific data. |
| `ShiftInputWindow(Tensor<>,Tensor<>)` | Shifts input window by appending new prediction. |
| `Train(Tensor<>,Tensor<>)` | Trains the STGNN model on a batch of input-target pairs. |
| `UpdateParameters(Vector<>)` | Updates parameters using the provided gradients. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers provided by the user. |

