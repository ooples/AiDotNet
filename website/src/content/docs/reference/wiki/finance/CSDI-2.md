---
title: "CSDI<T>"
description: "CSDI (Conditional Score-based Diffusion model for Imputation) for probabilistic time series imputation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Probabilistic`

CSDI (Conditional Score-based Diffusion model for Imputation) for probabilistic time series imputation.

## For Beginners

CSDI solves a common real-world problem: missing data.
Instead of simple interpolation, it generates plausible values that are consistent
with observed data.

**The Core Problem:**
Real-world time series often have missing values:

- Sensor failures
- Data transmission errors
- Irregular sampling
- Intentional gaps (surveys, experiments)

Simple methods (mean imputation, linear interpolation) ignore uncertainty.
CSDI gives you the FULL probability distribution of missing values.

**How Score-based Diffusion Works:**

1. **Score Matching:** Learn the gradient of log probability (the "score")
2. **Conditional Generation:** Keep observed values fixed, only modify missing ones
3. **Reverse SDE:** Follow the score to transform noise into realistic values
4. **Multiple Samples:** Generate diverse imputations for uncertainty

**CSDI Architecture:**

- Input: Values + Mask (indicating observed vs missing)
- Transformer blocks: Capture temporal and cross-feature dependencies
- Residual blocks: Process diffusion timestep and predict noise
- Conditional sampling: Only denoise missing positions

**Key Benefits:**

- Handles ANY missing pattern (not just regular gaps)
- Uncertainty quantification for imputed values
- Captures complex dependencies across time and features
- State-of-the-art imputation quality

## How It Works

CSDI is a probabilistic model for time series imputation that uses score-based diffusion
to fill in missing values with well-calibrated uncertainty estimates.

**Reference:** Tashiro et al., "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation", 2021.
https://arxiv.org/abs/2107.03502

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CSDI(NeuralNetworkArchitecture<>,CSDIOptions<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the CSDI model in native mode for training. |
| `CSDI(NeuralNetworkArchitecture<>,String,CSDIOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the CSDI model in ONNX mode for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDimension` | Gets the hidden dimension of the score network. |
| `IsChannelIndependent` |  |
| `NumDiffusionSteps` | Gets the number of diffusion steps. |
| `NumFeatures` |  |
| `NumSamples` | Gets the number of samples to generate for uncertainty estimation. |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` | Gets whether the model supports training (native mode only). |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddNoiseConditional(Tensor<>,Tensor<>,Int32)` | Adds noise to data, only affecting masked (missing) positions. |
| `ApplyInstanceNormalization(Tensor<>)` | Applies instance normalization to the input tensor (RevIN for non-stationarity). |
| `ApplyMask(Tensor<>,Tensor<>)` | Applies mask to tensor, zeroing out observed positions for loss computation. |
| `AutoregressiveForecast(Tensor<>,Int32)` | Performs autoregressive imputation (not typically used for CSDI). |
| `CombineDataAndMask(Tensor<>,Tensor<>,Int32)` | Combines data tensor with mask for network input. |
| `ComputePredictionIntervals(List<Tensor<>>,Double)` | Computes prediction intervals from samples. |
| `ComputeQuantiles(List<Tensor<>>,Double[])` | Computes quantiles from a list of samples. |
| `CreateNewInstance` | Creates a new instance of the CSDI model with the same configuration. |
| `DenoisingStepConditional(Tensor<>,Tensor<>,Tensor<>,Int32)` | Performs one step of the reverse diffusion process, only updating missing positions. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes CSDI-specific data when loading a saved model. |
| `Dispose(Boolean)` | Disposes of resources used by the CSDI model. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates imputation quality against actual values. |
| `ExtractLayerReferences` | Extracts references to key layers for efficient access during imputation. |
| `FlattenInput(Tensor<>)` | Flattens the input tensor for processing through dense layers. |
| `Forecast(Tensor<>,Double[])` | Generates imputed values for the given time series with missing data. |
| `ForecastWithIntervals(Tensor<>,Double)` | Generates imputations with prediction intervals for uncertainty quantification. |
| `Forward(Tensor<>)` | Performs the forward pass through all layers. |
| `ForwardForTraining(Tensor<>)` | Trains the CSDI model on a batch of complete data (simulating missing values). |
| `GenerateRandomMask(Int32[])` | Generates a random mask simulating missing data patterns. |
| `GenerateSamples(Tensor<>,Int32)` | Generates multiple imputation samples for uncertainty estimation. |
| `GetFinancialMetrics` | Gets financial-specific metrics about the model. |
| `GetModelMetadata` | Gets metadata about the CSDI model. |
| `GetOptions` |  |
| `ImputeNative(Tensor<>)` | Imputes missing values using native mode (full diffusion sampling). |
| `ImputeOnnx(Tensor<>)` | Imputes missing values using ONNX mode. |
| `InitializeDiffusionSchedule(Int32,Double,Double,String)` | Initializes the diffusion noise schedule for score-based sampling. |
| `InitializeLayers` | Initializes all layers for the CSDI model. |
| `InitializeWithNoise(Tensor<>,Tensor<>)` | Initializes missing positions with random noise, keeping observed values. |
| `ParseInputWithMask(Tensor<>)` | Parses input tensor that contains both data and mask. |
| `PredictCore(Tensor<>)` | Performs forward prediction (imputation) on the input tensor. |
| `PreserveObservedValues(Tensor<>,Tensor<>,Tensor<>)` | Ensures observed values are exactly preserved after imputation. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes CSDI-specific data for model persistence. |
| `UpdateParameters(Vector<>)` | Updates the model parameters using the optimizer (required override). |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers provided through the architecture. |

