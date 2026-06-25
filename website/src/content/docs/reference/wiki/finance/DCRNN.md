---
title: "DCRNN<T>"
description: "DCRNN (Diffusion Convolutional Recurrent Neural Network) for spatial-temporal forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Graph`

DCRNN (Diffusion Convolutional Recurrent Neural Network) for spatial-temporal forecasting.

## For Beginners

DCRNN was a breakthrough model for traffic prediction that introduced
two key innovations:

**The Key Insight:**
Traffic flow can be modeled as a diffusion process - congestion spreads through a road network
similar to how heat diffuses through a material. DCRNN captures this with diffusion convolution
while using an encoder-decoder architecture for multi-step forecasting.

**What Makes DCRNN Special:**

1. **Diffusion Convolution:** Models spatial dependencies as bidirectional random walks on the graph
2. **DCGRU Cells:** GRU cells where matrix multiplications are replaced with diffusion convolution
3. **Encoder-Decoder:** Seq2seq architecture for multi-step prediction
4. **Scheduled Sampling:** Gradually transitions from teacher forcing to autoregressive during training

**Mathematical Foundation:**
Diffusion convolution: X_star = sum_k (theta_k * (D_O^(-1)*W)^k * X + theta'_k * (D_I^(-1)*W^T)^k * X)
where D_O, D_I are out/in-degree matrices and W is the adjacency matrix.

**Architecture:**

- Encoder: Stacked DCGRU layers process input sequence
- Final encoder state becomes initial decoder state
- Decoder: Stacked DCGRU layers generate output autoregressively
- Output: Linear projection to forecast dimension

**Key Benefits:**

- Captures bidirectional spatial dependencies (upstream and downstream effects)
- Multi-step prediction without error accumulation from teacher forcing
- Effective on large-scale traffic networks

## How It Works

DCRNN combines diffusion convolution with sequence-to-sequence architecture for
traffic forecasting on road networks and other spatial-temporal prediction tasks.

**Reference:** Li et al., "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting", ICLR 2018.
https://arxiv.org/abs/1707.01926

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DCRNN(NeuralNetworkArchitecture<>,DCRNNOptions<>,Double[0:,0:],IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of DCRNN in native training/inference mode. |
| `DCRNN(NeuralNetworkArchitecture<>,String,DCRNNOptions<>,Double[0:,0:],IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of DCRNN in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DiffusionSteps` | Gets the number of diffusion steps. |
| `ForecastHorizon` | Gets the forecast horizon. |
| `IsChannelIndependent` |  |
| `NumFeatures` |  |
| `NumNodes` | Gets the number of nodes in the graph. |
| `NumSamples` | Gets the number of samples for uncertainty estimation. |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` | Gets whether the model supports training. |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDiffusionConvolution(Tensor<>)` | Applies diffusion convolution using precomputed diffusion matrices. |
| `ApplyInstanceNormalization(Tensor<>)` | Applies instance normalization (identity for DCRNN). |
| `ApplyMatrixToTensor(Double[0:,0:],Tensor<>,Int32,Int32)` | Applies a matrix to a tensor along the node dimension. |
| `AutoregressiveForecast(Tensor<>,Int32)` | Performs autoregressive forecasting for extended horizons. |
| `BuildConstantMatrix(Double[0:,0:],Int32)` | Builds a constant (non-trainable) `Tensor` from an n x n transition/adjacency matrix so it can participate in tape-aware matmuls. |
| `ComputeDiffusionPowers` | Precomputes powers of the diffusion matrices. |
| `ComputePredictionIntervals(List<Tensor<>>,Double)` | Computes prediction intervals from samples. |
| `ComputeQuantiles(List<Tensor<>>,Double[])` | Computes quantiles from samples. |
| `ConcatenatePredictions(List<Tensor<>>)` | Concatenates multiple prediction tensors. |
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes DCRNN-specific data. |
| `Dispose(Boolean)` | Releases resources used by the DCRNN model. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates forecast quality against actual values. |
| `ExtractLayerReferences` | Extracts references to specific layers for direct access during forward pass. |
| `FlattenInput(Tensor<>)` | Flattens input tensor for processing. |
| `Forecast(Tensor<>,Double[])` | Generates forecasts for all nodes. |
| `ForecastNative(Tensor<>)` | Performs native forward pass through DCRNN. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX inference. |
| `ForecastWithIntervals(Tensor<>,Double)` | Generates forecasts with prediction intervals. |
| `Forward(Tensor<>)` | Performs forward pass through all layers. |
| `GenerateSamples(Tensor<>,Int32)` | Generates multiple samples using MC dropout. |
| `GetFinancialMetrics` | Gets financial-specific metrics. |
| `GetModelMetadata` | Gets metadata about the DCRNN model. |
| `GetOptions` |  |
| `GetTeacherForcingRatio` | Gets the teacher forcing ratio based on scheduled sampling. |
| `InitializeDefaultDiffusionMatrices` | Initializes default diffusion matrices when no adjacency is provided. |
| `InitializeDiffusionMatrices(Double[0:,0:])` | Initializes diffusion matrices from the adjacency matrix. |
| `InitializeLayers` | Initializes the neural network layers for DCRNN. |
| `MatrixMultiply(Double[0:,0:],Double[0:,0:])` | Multiplies two matrices. |
| `PredictCore(Tensor<>)` | Makes a prediction using the DCRNN model. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes DCRNN-specific data. |
| `ShiftInputWindow(Tensor<>,Tensor<>)` | Shifts input window to include new prediction. |
| `Train(Tensor<>,Tensor<>)` | Trains the DCRNN model on provided data. |
| `UpdateParameters(Vector<>)` | Updates parameters using the provided gradients. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers provided by the user. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_backwardDiffusion` | Executes D_I^ for the DCRNN. |
| `_forwardDiffusion` | Executes D_O^ for the DCRNN. |

