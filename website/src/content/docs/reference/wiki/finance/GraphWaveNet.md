---
title: "GraphWaveNet<T>"
description: "GraphWaveNet (Graph WaveNet) for deep spatial-temporal graph modeling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Graph`

GraphWaveNet (Graph WaveNet) for deep spatial-temporal graph modeling.

## For Beginners

GraphWaveNet achieves top performance on traffic forecasting by combining:

**The Key Insight:**
Traditional methods either use a fixed graph structure OR learn it separately from forecasting.
GraphWaveNet jointly learns the graph structure AND the forecasting model, allowing them to
inform each other during training.

**What Problems Does GraphWaveNet Solve?**

- Traffic speed/flow prediction on road networks
- Air quality forecasting across sensor networks
- Electricity demand prediction across power grids
- Any spatial-temporal forecasting with underlying graph structure

**How GraphWaveNet Works:**

1. **Adaptive Graph:** Learns A = softmax(ReLU(E1 * E2^T)) from node embeddings
2. **Diffusion Conv:** H' = sum_k(P_f^k * H * W_k + P_b^k * H * V_k) for bidirectional propagation
3. **Gated TCN:** tanh(conv_f) ⊙ sigmoid(conv_g) with exponentially increasing dilation
4. **Skip Connections:** Sum outputs from all layers for multi-scale features

**GraphWaveNet Architecture:**

- Node Embeddings E1, E2: Learnable [num_nodes, embedding_dim] matrices
- Diffusion Convolution: Forward (A) and backward (A^T) random walk diffusion
- Gated TCN: Filter-Gate mechanism with dilated causal convolutions
- Skip Connections: Residual learning + skip from each layer to output
- Output: ReLU + Linear projection to forecast dimension

**Key Benefits:**

- Learns graph structure without prior knowledge
- Captures complex spatial dependencies via bidirectional diffusion
- Efficient training via parallel dilated convolutions
- State-of-the-art on METR-LA and PEMS-BAY traffic datasets

## How It Works

GraphWaveNet combines adaptive graph learning with WaveNet-style dilated causal
convolutions for state-of-the-art traffic and time series forecasting.

**Reference:** Wu et al., "Graph WaveNet for Deep Spatial-Temporal Graph Modeling", IJCAI 2019.
https://arxiv.org/abs/1906.00121

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphWaveNet(NeuralNetworkArchitecture<>,GraphWaveNetOptions<>,Double[0:,0:],IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the GraphWaveNet model in native mode for training. |
| `GraphWaveNet(NeuralNetworkArchitecture<>,String,GraphWaveNetOptions<>,Double[0:,0:],IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the GraphWaveNet model in ONNX mode for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DiffusionSteps` | Gets the number of diffusion steps (K). |
| `ForecastHorizon` | Gets the forecast horizon. |
| `IsChannelIndependent` |  |
| `LearnedAdjacency` | Gets the learned adaptive adjacency matrix. |
| `NumBlocks` | Gets the number of WaveNet blocks. |
| `NumFeatures` |  |
| `NumNodes` | Gets the number of nodes in the graph. |
| `NumSamples` | Gets the number of samples for uncertainty estimation. |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` | Gets whether the model supports training. |
| `UseAdaptiveGraph` | Gets whether adaptive graph learning is enabled. |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddSmallPerturbation(Tensor<>)` | Executes AddSmallPerturbation for the GraphWaveNet. |
| `ApplyDiffusionConvolution(Tensor<>)` | Applies diffusion convolution with forward and backward random walks. |
| `ApplyInstanceNormalization(Tensor<>)` | Applies instance normalization (identity for GraphWaveNet). |
| `AutoregressiveForecast(Tensor<>,Int32)` | Performs autoregressive forecasting. |
| `BuildConstantMatrix(Double[0:,0:])` | Builds a constant (non-trainable) `Tensor` from a graph support matrix so it can participate in tape-aware matmuls without being treated as a learnable parameter. |
| `ComputePredictionIntervals(List<Tensor<>>,Double)` | Executes private for the GraphWaveNet. |
| `ComputeQuantiles(List<Tensor<>>,Double[])` | Executes ComputeQuantiles for the GraphWaveNet. |
| `ConcatenatePredictions(List<Tensor<>>)` | Executes ConcatenatePredictions for the GraphWaveNet. |
| `CreateDefaultAdjacencyMatrix(Int32)` | Creates a default adjacency matrix with proximity-based connections. |
| `CreateNewInstance` | Creates a new instance. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes model-specific data. |
| `Dispose(Boolean)` | Executes Dispose for the GraphWaveNet. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates forecast quality. |
| `ExtractLayerReferences` | Extracts references to key layers for efficient access. |
| `FlattenInput(Tensor<>)` | Flattens input tensor. |
| `Forecast(Tensor<>,Double[])` | Generates forecasts. |
| `ForecastNative(Tensor<>)` | Executes ForecastNative for the GraphWaveNet. |
| `ForecastOnnx(Tensor<>)` | Executes ForecastOnnx for the GraphWaveNet. |
| `ForecastWithIntervals(Tensor<>,Double)` | Generates forecasts with prediction intervals. |
| `Forward(Tensor<>)` | Performs the forward pass. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward: calls `Tensor{` directly so dropout and the adaptive-graph conv stay in training mode under the tape. |
| `GenerateSamples(Tensor<>,Int32)` | Executes GenerateSamples for the GraphWaveNet. |
| `GetFinancialMetrics` | Gets financial metrics. |
| `GetModelMetadata` | Gets model metadata. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes all layers for the GraphWaveNet model. |
| `InitializeNodeEmbeddings` | Initializes the learnable node embeddings for adaptive graph learning. |
| `PredictCore(Tensor<>)` | Performs forward prediction. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes model-specific data. |
| `ShiftInputWindow(Tensor<>,Tensor<>)` | Executes ShiftInputWindow for the GraphWaveNet. |
| `Train(Tensor<>,Tensor<>)` | Trains the model. |
| `TransposeMatrix(Double[0:,0:])` | Transposes a matrix. |
| `UpdateAdaptiveAdjacency` | Updates the adaptive adjacency matrix from current node embeddings. |
| `UpdateNodeEmbeddingsFromGradient(Vector<>)` | Updates node embeddings based on loss gradient. |
| `UpdateParameters(Vector<>)` | Updates parameters. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers. |

