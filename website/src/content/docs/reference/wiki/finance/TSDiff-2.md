---
title: "TSDiff<T>"
description: "TSDiff (Time Series Diffusion) for probabilistic time series forecasting with self-guided diffusion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Probabilistic`

TSDiff (Time Series Diffusion) for probabilistic time series forecasting with self-guided diffusion.

## For Beginners

TSDiff takes a flexible approach to time series generation
by treating all tasks as conditional generation problems:

**The Key Innovation - Self-Guidance:**
Unlike models that need separate conditioning mechanisms, TSDiff uses its own
intermediate predictions to guide the generation process. This creates a "refinement loop"
where the model progressively improves its outputs.

**How Self-Guided Diffusion Works:**

1. **Initial Generation:** Start from noise, run standard diffusion
2. **Self-Prediction:** Use partial output to predict missing parts
3. **Guidance Gradient:** Compute gradient to improve consistency
4. **Refined Step:** Combine denoising step with guidance gradient
5. **Iterate:** Repeat until high-quality output

**TSDiff Architecture:**

- U-Net backbone with residual blocks for multi-scale processing
- Self-attention in bottleneck for long-range dependencies
- Timestep conditioning throughout the network
- Skip connections for preserving fine details

**Supported Tasks:**

- Forecasting: Condition on past, generate future
- Imputation: Condition on observed, generate missing
- Generation: Create synthetic time series from scratch

**Key Benefits:**

- Unified framework for multiple tasks
- Self-guidance improves temporal coherence
- Captures complex multivariate dynamics
- Produces diverse, high-quality samples

## How It Works

TSDiff is a versatile diffusion model that supports unconditional generation, forecasting,
and imputation through a unified self-guided diffusion framework.

**Reference:** Kollovieh et al., "Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting", 2023.
https://arxiv.org/abs/2307.11494

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TSDiff(NeuralNetworkArchitecture<>,String,TSDiffOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the TSDiff model in ONNX mode for inference. |
| `TSDiff(NeuralNetworkArchitecture<>,TSDiffOptions<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the TSDiff model in native mode for training. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets the context length (input history length). |
| `ForecastHorizon` | Gets the forecast horizon. |
| `GuidanceScale` | Gets the guidance scale for conditional generation. |
| `IsChannelIndependent` |  |
| `NumDiffusionSteps` | Gets the number of diffusion steps. |
| `NumFeatures` |  |
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
| `AddNoise(Tensor<>,Int32)` | Adds noise to data at a specific diffusion timestep. |
| `ApplyInstanceNormalization(Tensor<>)` | Applies instance normalization (RevIN). |
| `ApplySelfGuidance(Tensor<>,Tensor<>,Int32)` | Applies self-guidance to refine the current sample. |
| `AutoregressiveForecast(Tensor<>,Int32)` | Performs autoregressive forecasting. |
| `CombineContextAndSample(Tensor<>,Tensor<>)` | Combines context and sample for network input. |
| `ComputePredictionIntervals(List<Tensor<>>,Double)` | Computes prediction intervals from samples. |
| `ComputeQuantiles(List<Tensor<>>,Double[])` | Computes quantiles from samples. |
| `ConcatenatePredictions(List<Tensor<>>)` | Concatenates multiple predictions into one tensor. |
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DenoisingStep(Tensor<>,Tensor<>,Int32)` | Performs one step of the reverse diffusion process. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes TSDiff-specific data. |
| `Dispose(Boolean)` | Disposes of resources. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates forecast quality against actual values. |
| `ExtractLayerReferences` | Extracts references to key layers for efficient access. |
| `FlattenInput(Tensor<>)` | Flattens input tensor for dense layer processing. |
| `Forecast(Tensor<>,Double[])` | Generates forecasts for the given input time series. |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting with self-guided diffusion. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `ForecastWithIntervals(Tensor<>,Double)` | Generates forecasts with prediction intervals. |
| `Forward(Tensor<>)` | Performs the forward pass through all layers. |
| `ForwardForTraining(Tensor<>)` | Trains the TSDiff model on a batch of input-target pairs. |
| `GenerateNoise(Int32)` | Generates random noise for initialization. |
| `GenerateSamples(Tensor<>,Int32)` | Generates multiple forecast samples for uncertainty estimation. |
| `GetFinancialMetrics` | Gets financial-specific metrics. |
| `GetModelMetadata` | Gets metadata about the TSDiff model. |
| `GetOptions` |  |
| `InitializeDiffusionSchedule(Int32,Double,Double,String)` | Initializes the diffusion noise schedule. |
| `InitializeLayers` | Initializes all layers for the TSDiff model. |
| `PredictCore(Tensor<>)` | Performs forward prediction on the input tensor. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes TSDiff-specific data. |
| `ShiftInputWindow(Tensor<>,Tensor<>)` | Shifts input window by appending new prediction. |
| `UpdateParameters(Vector<>)` | Updates parameters (required override). |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers. |

