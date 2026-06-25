---
title: "Hippo<T>"
description: "HiPPO (High-order Polynomial Projection Operators) for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.StateSpace`

HiPPO (High-order Polynomial Projection Operators) for time series forecasting.

## For Beginners

HiPPO answers a fundamental question in sequence modeling:
"How do we optimally remember a continuous history in a fixed-size memory?"

**The Core Problem:**
When processing a sequence (like a time series), we need to "remember" the past.
But we can't store infinite history - we need a fixed-size "state".
HiPPO shows how to create a state that is the OPTIMAL approximation of history.

**How It Works:**

1. **Polynomial Basis:** Approximate history using polynomials (like Legendre)
2. **Optimal Projection:** State x(t) = coefficients of best polynomial fit to history
3. **Online Update:** Update state efficiently as new inputs arrive
4. **Memory Matrix A:** Derived mathematically to ensure optimal approximation

**The Math (simplified):**

- State Space Model: dx/dt = Ax + Bu
- A is the "HiPPO matrix" for your chosen polynomial basis
- x(t) contains polynomial coefficients: history(s) ≈ Σ x_i(t) * P_i(s)
- Different A matrices give different memory properties

**Available Methods:**

- HiPPO-LegS: Sliding window memory (recent history weighted equally)
- HiPPO-LegT: Fixed window [0,t] (entire history weighted equally)
- HiPPO-LagT: Exponential decay (recent > distant)

**Why HiPPO Matters:**

- Mathematically principled initialization for SSMs
- Provably optimal history compression
- Foundation for S4, Mamba, and modern sequence models
- Enables models to handle very long sequences efficiently

## How It Works

HiPPO provides the theoretical foundation for efficient state space models like S4 and Mamba.
It defines optimal state matrices for compressing sequential input history into a fixed-size state.

**Reference:** Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial Projections", 2020.
https://arxiv.org/abs/2008.07669

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Hippo(NeuralNetworkArchitecture<>,HippoOptions<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the HiPPO model in native mode for training. |
| `Hippo(NeuralNetworkArchitecture<>,String,HippoOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the HiPPO model in ONNX mode for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets the input context length for the model. |
| `ForecastHorizon` | Gets the forecast horizon (number of future steps to predict). |
| `HippoMethod` | Gets the HiPPO method used for state initialization. |
| `IsChannelIndependent` |  |
| `NumFeatures` |  |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `StateDimension` | Gets the state dimension (polynomial order) of the HiPPO model. |
| `Stride` |  |
| `SupportsTraining` | Gets whether the model supports training (native mode only). |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` | Applies instance normalization to the input. |
| `AutoregressiveForecast(Tensor<>,Int32)` | Performs autoregressive forecasting step by step. |
| `ComputeHippoBVector(Int32,String)` | Computes the B vector for the HiPPO model. |
| `ComputeHippoMatrix(Int32,String)` | Computes the HiPPO matrix for a given method. |
| `ComputePredictionIntervals(List<Tensor<>>,Double)` | Computes prediction intervals from Monte Carlo samples. |
| `ConcatenatePredictions(List<Tensor<>>)` | Concatenates multiple prediction tensors into a single tensor. |
| `CreateNewInstance` | Creates a new instance of the HiPPO model with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes HiPPO-specific data when loading a saved model. |
| `Dispose(Boolean)` | Disposes of managed and unmanaged resources. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates forecast quality against actual values. |
| `ExtractLayerReferences` | Extracts references to key layers for efficient access. |
| `FlattenInput(Tensor<>)` | Flattens the input tensor for processing through dense layers. |
| `Forecast(Tensor<>,Double[])` | Generates forecasts for the given input time series. |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting through the layer stack. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting using the pretrained model. |
| `ForecastWithIntervals(Tensor<>,Double)` | Generates forecasts with prediction intervals for uncertainty quantification. |
| `Forward(Tensor<>)` | Performs the forward pass through all layers. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward: calls `Tensor{` directly so the tape sees the Hippo SSM layers in training mode. |
| `GetFinancialMetrics` | Gets financial-specific metrics about the model. |
| `GetModelMetadata` | Gets metadata about the HiPPO model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes all layers for the HiPPO model. |
| `PredictCore(Tensor<>)` | Performs forward prediction on the input tensor. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes HiPPO-specific data for model persistence. |
| `ShiftInputWindow(Tensor<>,Tensor<>)` | Shifts the input window by removing oldest values and appending new prediction. |
| `Train(Tensor<>,Tensor<>)` | Trains the HiPPO model on a batch of input-target pairs. |
| `UpdateParameters(Vector<>)` | Updates the model parameters using the optimizer (required override). |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers provided through the architecture. |

