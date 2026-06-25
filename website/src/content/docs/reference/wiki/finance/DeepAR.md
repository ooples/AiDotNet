---
title: "DeepAR<T>"
description: "DeepAR probabilistic autoregressive forecasting model using LSTM networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Neural`

DeepAR probabilistic autoregressive forecasting model using LSTM networks.

## For Beginners

DeepAR is special because it doesn't just predict a single value - it
predicts a probability distribution. This means you get:

- A most likely value (the mean)
- A measure of confidence (the standard deviation)
- The ability to generate prediction intervals (e.g., 95% confidence bounds)

Key features:

- **Autoregressive:** Each prediction depends on previous predictions
- **Probabilistic:** Outputs full distributions, not just point forecasts
- **Multi-series:** Can learn patterns across many related time series
- **Covariates:** Can include additional features like holidays or promotions

## How It Works

DeepAR is a probabilistic forecasting model that produces forecast distributions rather than
point predictions. It uses autoregressive recurrent neural networks to learn temporal patterns
and outputs distribution parameters (e.g., mean and standard deviation for Gaussian).

**Reference:** Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive
Recurrent Networks", International Journal of Forecasting 2020.
https://arxiv.org/abs/1704.04110

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepAR` | Creates a DeepAR model with default configuration for native training. |
| `DeepAR(NeuralNetworkArchitecture<>,DeepAROptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a DeepAR network in native mode for training from scratch. |
| `DeepAR(NeuralNetworkArchitecture<>,String,DeepAROptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a DeepAR network using pretrained ONNX model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` | Gets whether the model processes channels independently. |
| `PatchSize` | Gets the patch size for the model. |
| `Stride` | Gets the stride for the model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` | Applies scaling to the input tensor for DeepAR processing. |
| `ApplyScaling(Tensor<>)` | Applies scaling by dividing by mean absolute value. |
| `AutoregressiveForecast(Tensor<>,Int32)` | Generates multi-step forecasts by feeding predictions back into the model. |
| `ComputeCRPS(Tensor<>,Tensor<>)` | Computes CRPS (Continuous Ranked Probability Score) for probabilistic evaluation. |
| `ComputeGradient(Tensor<>,Tensor<>)` | Computes gradient for backpropagation. |
| `ComputeNegativeLogLikelihood(Tensor<>,Tensor<>)` | Computes negative log-likelihood loss for the distribution. |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple predictions into a single tensor. |
| `CreateNewInstance` | Creates a new instance of this model with the same configuration. |
| `DeserializeModelSpecificData(BinaryReader)` | Reads DeepAR-specific configuration during deserialization. |
| `Dispose(Boolean)` | Releases resources used by the DeepAR model. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates forecast accuracy using common error metrics. |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection. |
| `Forecast(Tensor<>,Double[])` | Generates probabilistic forecasts for the given input data. |
| `ForecastNative(Tensor<>,Double[])` | Performs native mode forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through the DeepAR network. |
| `ForwardNativeForTraining(Tensor<>)` | Tape-connected forward used by `Tensor{`. |
| `GetFinancialMetrics` | Gets metrics specific to the DeepAR model configuration. |
| `GetModelMetadata` | Gets metadata about the model for serialization and inspection. |
| `GetOptions` |  |
| `GetStandardNormalQuantile(Double)` | Computes the standard normal quantile (inverse CDF). |
| `InitializeLayers` | Initializes the neural network layers for DeepAR. |
| `ReverseScaling(Tensor<>)` | Reverses the scaling applied during preprocessing. |
| `SampleQuantiles(Tensor<>,Double[])` | Samples quantiles from the forecast distribution. |
| `SerializeModelSpecificData(BinaryWriter)` | Writes DeepAR-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input by incorporating recent predictions. |
| `TrainCore(Tensor<>,Tensor<>,Tensor<>)` | Trains the model on a single batch of input-output pairs. |
| `UpdateParameters(Vector<>)` | Updates the model's parameters using the provided gradients. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet DeepAR's architectural requirements. |
| `ValidateInputShape(Tensor<>)` | Validates the input tensor shape for DeepAR. |
| `ValidateOptions(DeepAROptions<>)` | Validates the DeepAR options. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_distributionType` | The output distribution type (gaussian, negative_binomial, student_t). |
| `_dropout` | The dropout rate for regularization. |
| `_embeddingDim` | Embedding dimension for categorical features. |
| `_hiddenSize` | The hidden size of LSTM cells. |
| `_inputProjection` | Input projection layer to prepare features for LSTM processing. |
| `_lastSigma` | The last computed sigma from Forward (used for quantile sampling). |
| `_layerNorm` | Layer normalization for stable training. |
| `_lstmLayers` | Stacked LSTM layers for sequence modeling. |
| `_muProjection` | Output layer for distribution mean (mu). |
| `_numLstmLayers` | The number of stacked LSTM layers. |
| `_numSamples` | Number of samples for Monte Carlo estimation. |
| `_optimizer` | The optimizer for training. |
| `_random` | Random number generator for sampling. |
| `_scaleStd` | Instance normalization scale for denormalization. |
| `_sigmaProjection` | Output layer for distribution scale (sigma). |
| `_useScaling` | Whether to use scaling (dividing by mean absolute value). |

