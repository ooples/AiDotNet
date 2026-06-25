---
title: "TimeGrad<T>"
description: "TimeGrad (Autoregressive Denoising Diffusion Model) for probabilistic time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Probabilistic`

TimeGrad (Autoregressive Denoising Diffusion Model) for probabilistic time series forecasting.

## For Beginners

TimeGrad brings the power of diffusion models (like DALL-E
for images) to time series forecasting:

**The Core Problem:**
Most forecasting models give ONE prediction. But you often need to know:

- How confident is this prediction?
- What's the worst-case scenario?
- What range of outcomes is possible?

TimeGrad solves this by generating MANY possible future paths, giving you
a full probability distribution of what might happen.

**How Diffusion Works:**

1. **Forward Process:** Gradually add noise to real data until it's pure noise
2. **Reverse Process:** Learn to remove noise step-by-step
3. **Conditioning:** Use historical data to guide the denoising
4. **Sampling:** Start from noise, denoise to get realistic forecasts

**TimeGrad Architecture:**

- RNN encoder: Processes historical data into a hidden state
- Denoising network: Predicts and removes noise, conditioned on RNN state
- Sampling: Run reverse process multiple times for uncertainty

**Key Benefits:**

- Probabilistic forecasts with uncertainty quantification
- Well-calibrated prediction intervals
- Can generate diverse, realistic scenarios
- Captures complex multimodal distributions

## How It Works

TimeGrad is a probabilistic time series forecasting model that uses denoising diffusion
to generate accurate forecasts with well-calibrated uncertainty estimates.

**Reference:** Rasul et al., "Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting", 2021.
https://arxiv.org/abs/2101.12072

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeGrad(NeuralNetworkArchitecture<>,String,TimeGradOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the TimeGrad model in ONNX mode for inference. |
| `TimeGrad(NeuralNetworkArchitecture<>,TimeGradOptions<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the TimeGrad model in native mode for training. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets the input context length for the model. |
| `ForecastHorizon` | Gets the forecast horizon (number of future steps to predict). |
| `IsChannelIndependent` |  |
| `NumDiffusionSteps` | Gets the number of diffusion steps. |
| `NumFeatures` |  |
| `NumSamples` | Gets the number of samples to generate for probabilistic forecasting. |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` | Gets whether the model supports training (native mode only). |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddNoise(Tensor<>,Int32)` | Adds noise to data using the forward diffusion process. |
| `ApplyInstanceNormalization(Tensor<>)` | Applies instance normalization to the input. |
| `AutoregressiveForecast(Tensor<>,Int32)` | Performs autoregressive forecasting step by step. |
| `CombineContextAndNoisy(Tensor<>,Tensor<>,Int32)` | Combines context and noisy target for the denoising network. |
| `ComputePredictionIntervals(List<Tensor<>>,Double)` | Computes prediction intervals from samples. |
| `ComputeQuantiles(List<Tensor<>>,Double[])` | Computes quantiles from samples. |
| `ConcatenatePredictions(List<Tensor<>>)` | Concatenates multiple prediction tensors. |
| `CreateNewInstance` | Creates a new instance of the TimeGrad model with the same configuration. |
| `DenoisingStep(Tensor<>,Tensor<>,Int32)` | Performs one denoising step in the reverse diffusion process. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes TimeGrad-specific data when loading a saved model. |
| `Dispose(Boolean)` | Disposes of managed and unmanaged resources. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates forecast quality against actual values. |
| `ExtractLayerReferences` | Extracts references to key layers for efficient access. |
| `FlattenInput(Tensor<>)` | Flattens the input tensor for processing through dense layers. |
| `Forecast(Tensor<>,Double[])` | Generates forecasts for the given input time series. |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting using reverse diffusion. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting using the pretrained model. |
| `ForecastWithIntervals(Tensor<>,Double)` | Generates forecasts with prediction intervals for uncertainty quantification. |
| `Forward(Tensor<>)` | Performs the forward pass through all layers. |
| `ForwardForTraining(Tensor<>)` | Trains the TimeGrad model on a batch of input-target pairs. |
| `GenerateNoise(Int32)` | Generates random Gaussian noise. |
| `GenerateSamples(Tensor<>,Int32)` | Generates multiple forecast samples using diffusion. |
| `GetFinancialMetrics` | Gets financial-specific metrics about the model. |
| `GetModelMetadata` | Gets metadata about the TimeGrad model. |
| `GetOptions` |  |
| `InitializeDiffusionSchedule(Int32,Double,Double,String)` | Initializes the diffusion noise schedule. |
| `InitializeLayers` | Initializes all layers for the TimeGrad model. |
| `PredictCore(Tensor<>)` | Performs forward prediction on the input tensor. |
| `SampleGaussian` | Samples from a standard Gaussian distribution. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes TimeGrad-specific data for model persistence. |
| `ShiftInputWindow(Tensor<>,Tensor<>)` | Shifts the input window for autoregressive forecasting. |
| `UpdateParameters(Vector<>)` | Updates the model parameters using the optimizer (required override). |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers provided through the architecture. |

