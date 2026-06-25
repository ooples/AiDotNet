---
title: "TimeGrad<T>"
description: "TimeGrad — Autoregressive Denoising Diffusion Model for Time Series Forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

TimeGrad — Autoregressive Denoising Diffusion Model for Time Series Forecasting.

## For Beginners

TimeGrad predicts time series step by step, where at each step
it uses a diffusion process to generate the next value. Think of it as a storyteller who
writes one sentence at a time, but for each sentence uses a careful drafting process to get
it right. By generating many possible futures, TimeGrad provides not just a single forecast
but a range of scenarios with probabilities, helping you understand how confident the
prediction is.

## How It Works

TimeGrad combines an autoregressive RNN with a conditional diffusion process for
probabilistic multi-step forecasting. It generates multiple forecast samples to
provide well-calibrated uncertainty estimates.

**Reference:** Rasul et al., "Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting", ICML 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeGrad(NeuralNetworkArchitecture<>,String,TimeGradOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TimeGrad model using a pretrained ONNX model. |
| `TimeGrad(NeuralNetworkArchitecture<>,TimeGradOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TimeGrad model in native mode for training. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
| `MaxContextLength` |  |
| `MaxPredictionHorizon` |  |
| `ModelSize` |  |
| `NumFeatures` |  |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` |  |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `ComputeNoiseSchedule` | Precomputes the DDPM noise schedule arrays for the diffusion process. |
| `ConcatenateForDenoising(Tensor<>,Tensor<>,Int32)` | Concatenates the noisy sample x_t with the RNN hidden state and diffusion timestep to form the input for the denoising network. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `Forecast(Tensor<>,Double[])` |  |
| `ForwardForTraining(Tensor<>)` | Tape-aware training forward. |
| `ForwardNative(Tensor<>)` | Performs inference via the DDPM reverse process (iterative denoising). |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SampleStandardNormal(Random)` | Samples from a standard normal distribution using Box-Muller transform. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `UpdateParameters(Vector<>)` |  |

