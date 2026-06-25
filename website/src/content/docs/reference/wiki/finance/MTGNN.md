---
title: "MTGNN<T>"
description: "MTGNN (Multivariate Time-series Graph Neural Network) for automatic graph learning and forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Graph`

MTGNN (Multivariate Time-series Graph Neural Network) for automatic graph learning and forecasting.

## For Beginners

MTGNN is unique because it LEARNS how variables are connected:

**The Key Insight:**
Unlike other graph models that require you to define the graph upfront, MTGNN
automatically discovers which time series influence each other. It learns node
embeddings whose similarities form an adaptive graph.

**What Problems Does MTGNN Solve?**

- Traffic prediction when road relationships are complex or unknown
- Multivariate financial forecasting with hidden correlations
- Sensor networks where dependencies change over time
- Any multivariate series where inter-variable relationships are important but unknown

**How MTGNN Works:**

1. **Graph Learning:** Learns node embeddings E1, E2; computes A = softmax(E1 * E2^T)
2. **Mix-hop Propagation:** Aggregates 1-hop, 2-hop, ... K-hop neighbors
3. **Dilated Inception:** Captures multi-scale temporal patterns via dilated convolutions
4. **Joint Learning:** Graph structure and predictions are optimized together

**MTGNN Architecture:**

- Node Embeddings: Two learnable embedding matrices E1, E2
- Adaptive Adjacency: A = softmax(ReLU(E1 * E2^T - E2 * E1^T))
- Mix-hop Propagation: H_out = concat(H, A*H, A^2*H, ..., A^K*H) * W
- Dilated Inception: Parallel convolutions with exponentially increasing dilation
- Skip Connections: Gated residual connections between layers

**Key Benefits:**

- No need to predefine graph structure
- Discovers hidden variable relationships automatically
- Captures both unidirectional and bidirectional dependencies
- Scales to hundreds of variables with subgraph sampling

## How It Works

MTGNN automatically discovers the graph structure from data while performing spatio-temporal
forecasting, eliminating the need for a predefined adjacency matrix.

**Reference:** Wu et al., "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks", KDD 2020.
https://arxiv.org/abs/2005.11650

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MTGNN(NeuralNetworkArchitecture<>,MTGNNOptions<>,Double[0:,0:],IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the MTGNN model in native mode for training. |
| `MTGNN(NeuralNetworkArchitecture<>,String,MTGNNOptions<>,Double[0:,0:],IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the MTGNN model in ONNX mode for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ForecastHorizon` | Gets the forecast horizon. |
| `IsChannelIndependent` |  |
| `LearnedAdjacency` | Gets the learned adaptive adjacency matrix (read-only copy). |
| `MixHopDepth` | Gets the mix-hop propagation depth. |
| `NodeEmbeddingDim` | Gets the node embedding dimension. |
| `NumFeatures` |  |
| `NumNodes` | Gets the number of nodes (variables/time series). |
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
| `ApplyInstanceNormalization(Tensor<>)` | Applies instance normalization (identity for MTGNN). |
| `ApplyMixHopPropagation(Tensor<>)` | Applies mix-hop propagation using the adaptive adjacency matrix. |
| `ApplyMixHopPropagationTape(Tensor<>)` | Tape-aware MixHop propagation: accumulates weighted powers of the adaptive adjacency matrix times the node-feature matrix (`H + sum_k w_k * A^k * H`) via `Tensor{`. |
| `AutoregressiveForecast(Tensor<>,Int32)` | Performs autoregressive forecasting for extended horizons. |
| `ComputePredictionIntervals(List<Tensor<>>,Double)` | Computes prediction intervals from samples. |
| `ComputeQuantiles(List<Tensor<>>,Double[])` | Computes quantiles from samples. |
| `ConcatenatePredictions(List<Tensor<>>)` | Concatenates multiple predictions. |
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes MTGNN-specific data. |
| `Dispose(Boolean)` | Disposes of resources used by the model. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates forecast quality against actual values. |
| `ExtractLayerReferences` | Extracts references to key layers for efficient access. |
| `FlattenInput(Tensor<>)` | Flattens input tensor for dense layer processing. |
| `Forecast(Tensor<>,Double[])` | Generates forecasts for all nodes. |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `ForecastWithIntervals(Tensor<>,Double)` | Generates forecasts with prediction intervals. |
| `Forward(Tensor<>)` | Performs the forward pass through all layers. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward: mirrors `Tensor{` but swaps `Tensor{` for `Tensor{`, which uses `Engine.TensorMatMul` instead of the tape-breaking ToVector/manual-loop accumulator. |
| `GenerateSamples(Tensor<>,Int32)` | Generates multiple forecast samples using MC Dropout. |
| `GetFinancialMetrics` | Gets financial-specific metrics. |
| `GetModelMetadata` | Gets metadata about the MTGNN model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes all layers for the MTGNN model. |
| `InitializeNodeEmbeddings` | Initializes the learnable node embeddings for adaptive graph learning. |
| `PredictCore(Tensor<>)` | Performs forward prediction on the input tensor. |
| `ReshapeToNodes(Tensor<>)` | Reshapes the input to [numNodes, featuresPerNode] (numNodes as the batch dimension) via the tape-aware Engine.Reshape so the per-node MLP layers apply SHARED weights across nodes and gradients flow back to the input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes MTGNN-specific data. |
| `ShiftInputWindow(Tensor<>,Tensor<>)` | Shifts input window by appending new prediction. |
| `Train(Tensor<>,Tensor<>)` | Trains the MTGNN model on a batch of input-target pairs. |
| `UpdateAdaptiveAdjacency` | Updates the adaptive adjacency matrix from current node embeddings. |
| `UpdateNodeEmbeddingsFromGradient(Vector<>)` | Updates node embeddings based on loss gradient. |
| `UpdateParameters(Vector<>)` | Updates parameters using the provided gradients. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers provided by the user. |

