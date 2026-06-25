---
title: "ScoreGrad<T>"
description: "ScoreGrad (Score-based Gradient Model) for probabilistic time series forecasting using score matching."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Probabilistic`

ScoreGrad (Score-based Gradient Model) for probabilistic time series forecasting using score matching.

## For Beginners

ScoreGrad uses a principled probabilistic approach called "score matching":

**The Key Insight:**
Instead of directly generating samples or predicting noise, ScoreGrad learns the "score function" -
the direction in which probability increases. By following this direction (with some randomness),
the model can generate realistic time series from noise.

**What is the Score Function?**
The score is ∇_x log p(x) - the gradient of log probability with respect to the data x.

- If you're at point x, the score tells you which direction has higher probability
- Following the score uphill leads to likely data points
- It's like having a compass that points toward "good" data

**How ScoreGrad Works:**

1. **Score Network:** Train a neural network to predict the score at different noise levels
2. **Denoising Score Matching:** Learn scores by adding noise and learning to denoise
3. **Langevin Dynamics:** Sample by iteratively following the score plus random noise
4. **Annealing:** Start with high noise (easy to sample), gradually reduce for refinement

**ScoreGrad Architecture:**

- Score Network: Predicts ∇_x log p(x|σ) conditioned on noise level σ
- Noise Embedding: Sinusoidal encoding of current noise level
- Residual Blocks: Deep architecture for learning complex score functions
- Output: Gradient direction at each position in the forecast

**Key Benefits:**

- Principled probabilistic foundation
- Flexible sampling (adjust step size and iterations)
- Natural uncertainty quantification
- Works well for complex multivariate dynamics

## How It Works

ScoreGrad is a score-based generative model that learns the gradient of the log probability
density function (score) and uses it for sampling via Langevin dynamics.

**Reference:** Yan et al., "ScoreGrad: Multivariate Probabilistic Time Series Forecasting with Continuous Energy-based Generative Models", 2021.
https://arxiv.org/abs/2106.10121

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ScoreGrad(NeuralNetworkArchitecture<>,ScoreGradOptions<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the ScoreGrad model in native mode for training. |
| `ScoreGrad(NeuralNetworkArchitecture<>,String,ScoreGradOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the ScoreGrad model in ONNX mode for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets the context length (input history length). |
| `ForecastHorizon` | Gets the forecast horizon. |
| `IsChannelIndependent` |  |
| `NumFeatures` |  |
| `NumLangevinSteps` | Gets the number of Langevin steps per noise level. |
| `NumNoiseScales` | Gets the number of noise scales for multi-scale score matching. |
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
| `AddNoise(Tensor<>,Double)` | Adds Gaussian noise to data with specified standard deviation. |
| `ApplyInstanceNormalization(Tensor<>)` | Applies instance normalization (identity for ScoreGrad). |
| `AutoregressiveForecast(Tensor<>,Int32)` | Performs autoregressive forecasting. |
| `CombineInputs(Tensor<>,Tensor<>,Tensor<>)` | Combines context, sample, and sigma embedding into network input. |
| `ComputePredictionIntervals(List<Tensor<>>,Double)` | Computes prediction intervals from samples. |
| `ComputeQuantiles(List<Tensor<>>,Double[])` | Computes quantiles from samples. |
| `ComputeTrueScore(Tensor<>,Double)` | Computes the true score for denoising score matching. |
| `ConcatenatePredictions(List<Tensor<>>)` | Concatenates multiple predictions into one tensor. |
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `CreateSigmaEmbedding(Double)` | Creates a sinusoidal embedding for the noise level (sigma). |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes ScoreGrad-specific data. |
| `Dispose(Boolean)` | Disposes of resources. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates forecast quality against actual values. |
| `ExtractLayerReferences` | Extracts references to key layers for efficient access. |
| `FlattenInput(Tensor<>)` | Flattens input tensor for dense layer processing. |
| `Forecast(Tensor<>,Double[])` | Generates forecasts for the given input time series. |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting with annealed Langevin dynamics. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `ForecastWithIntervals(Tensor<>,Double)` | Generates forecasts with prediction intervals. |
| `Forward(Tensor<>)` | Performs the forward pass through all layers. |
| `ForwardForTraining(Tensor<>)` | Trains the ScoreGrad model using denoising score matching. |
| `GenerateNoise(Int32,Double)` | Generates random Gaussian noise scaled by sigma. |
| `GenerateSamples(Tensor<>,Int32)` | Generates multiple forecast samples for uncertainty estimation. |
| `GetFinancialMetrics` | Gets financial-specific metrics. |
| `GetModelMetadata` | Gets metadata about the ScoreGrad model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes all layers for the ScoreGrad model. |
| `InitializeNoiseSchedule(Int32,Double,Double)` | Initializes the geometric noise schedule (sigma levels). |
| `LangevinStep(Tensor<>,Tensor<>,Double,Double)` | Performs one step of Langevin dynamics. |
| `PredictCore(Tensor<>)` | Performs forward prediction on the input tensor. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes ScoreGrad-specific data. |
| `ShiftInputWindow(Tensor<>,Tensor<>)` | Shifts input window by appending new prediction. |
| `UpdateParameters(Vector<>)` | Updates parameters using the provided gradients. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers provided by the user. |

