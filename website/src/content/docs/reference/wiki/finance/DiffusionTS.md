---
title: "DiffusionTS<T>"
description: "DiffusionTS (Interpretable Diffusion for Time Series) for probabilistic forecasting with seasonal-trend decomposition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Probabilistic`

DiffusionTS (Interpretable Diffusion for Time Series) for probabilistic forecasting with seasonal-trend decomposition.

## For Beginners

DiffusionTS makes diffusion models more interpretable
by decomposing time series into understandable components:

**The Key Insight:**
Time series often have clear structure (trends, seasonality) that gets lost in
"black box" models. DiffusionTS preserves this structure by generating each
component separately and combining them.

**How DiffusionTS Works:**

1. **Decomposition:** Split time series into trend, seasonal, and residual
2. **Component Diffusion:** Generate each component with specialized networks
3. **Reconstruction:** Combine components to form final forecast
4. **Interpretation:** Each component has clear meaning

**DiffusionTS Architecture:**

- Trend Network: Captures long-term movements (slow, smooth)
- Seasonal Network: Captures periodic patterns (daily, weekly, yearly)
- Residual Network: Captures irregular fluctuations
- Fusion Module: Combines components coherently

**Key Benefits:**

- Interpretable decomposition of forecasts
- Can enforce structural constraints (smooth trends, periodic seasons)
- Better uncertainty quantification per component
- Enables "what-if" analysis by modifying components

## How It Works

DiffusionTS is an interpretable diffusion model that uses seasonal-trend decomposition
to generate forecasts with clear interpretable components.

**Reference:** Yuan and Qiu, "Diffusion-TS: Interpretable Diffusion for General Time Series Generation", 2024.
https://arxiv.org/abs/2403.01742

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffusionTS(NeuralNetworkArchitecture<>,DiffusionTSOptions<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the DiffusionTS model in native mode for training. |
| `DiffusionTS(NeuralNetworkArchitecture<>,String,DiffusionTSOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the DiffusionTS model in ONNX mode for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DecompositionPeriod` | Gets the decomposition period for seasonal extraction. |
| `ForecastHorizon` | Gets the forecast horizon. |
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
| `UseSeasonalComponent` | Gets whether seasonal component is used. |
| `UseTrendComponent` | Gets whether trend component is used. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddNoise(Tensor<>,Int32)` | Adds noise to data at a specific diffusion timestep. |
| `ApplyInstanceNormalization(Tensor<>)` | Applies instance normalization (RevIN) to the input. |
| `ApplySmoothness(Tensor<>)` | Applies smoothness constraint to the trend component. |
| `AutoregressiveForecast(Tensor<>,Int32)` | Performs autoregressive forecasting for extended horizons. |
| `CombineComponents(Tensor<>,Tensor<>,Tensor<>)` | Combines trend, seasonal, and residual components into a single tensor. |
| `CombineTensors(Tensor<>,Tensor<>)` | Combines two tensors by concatenation. |
| `ComputePredictionIntervals(List<Tensor<>>,Double)` | Computes prediction intervals from multiple samples. |
| `ComputeQuantiles(List<Tensor<>>,Double[])` | Computes quantiles from multiple samples. |
| `ConcatenatePredictions(List<Tensor<>>)` | Concatenates multiple prediction tensors. |
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DecomposeTimeSeries(Tensor<>)` | Decomposes a time series into trend, seasonal, and residual components. |
| `DenoisingStep(Tensor<>,Tensor<>,Int32)` | Performs one step of the reverse diffusion process. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes DiffusionTS-specific data from a saved model. |
| `Dispose(Boolean)` | Disposes of managed and unmanaged resources. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates forecast quality against actual values. |
| `ExtractComponent(Tensor<>,Int32,Int32)` | Extracts a component from a combined tensor. |
| `ExtractLayerReferences` | Extracts references to key layers for efficient access during forward/backward passes. |
| `FlattenInput(Tensor<>)` | Flattens input tensor for dense layer processing. |
| `Forecast(Tensor<>,Double[])` | Generates forecasts for the given input time series. |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting with interpretable diffusion. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting using a pretrained model. |
| `ForecastWithIntervals(Tensor<>,Double)` | Generates forecasts with prediction intervals. |
| `Forward(Tensor<>)` | Performs the forward pass through all layers. |
| `ForwardForTraining(Tensor<>)` | Trains the DiffusionTS model on a batch of input-target pairs. |
| `GenerateNoise(Int32)` | Generates random Gaussian noise for initialization. |
| `GenerateSamples(Tensor<>,Int32)` | Generates multiple forecast samples for uncertainty estimation. |
| `GetFinancialMetrics` | Gets financial-specific metrics for model evaluation. |
| `GetModelMetadata` | Gets metadata about the DiffusionTS model. |
| `GetOptions` |  |
| `InitializeDiffusionSchedule(Int32,Double,Double,String)` | Initializes the diffusion noise schedule. |
| `InitializeLayers` | Initializes all layers for the DiffusionTS model. |
| `PredictCore(Tensor<>)` | Performs forward prediction on the input tensor. |
| `ReconstructFromComponents(Tensor<>,Tensor<>,Tensor<>)` | Reconstructs the time series from its decomposed components. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes DiffusionTS-specific data for model persistence. |
| `ShiftInputWindow(Tensor<>,Tensor<>)` | Shifts input window by appending new prediction. |
| `UpdateParameters(Vector<>)` | Updates parameters using the provided gradients. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers provided by the user. |

