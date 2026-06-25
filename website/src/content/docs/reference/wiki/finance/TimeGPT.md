---
title: "TimeGPT<T>"
description: "TimeGPT-style time series foundation model implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

TimeGPT-style time series foundation model implementation.

## For Beginners

TimeGPT brings GPT-like capabilities to time series:

**The Key Insight:**
Just as GPT was trained on internet-scale text data to become a general-purpose
language model, TimeGPT is trained on millions of diverse time series to become
a general-purpose forecasting model.

**Core Features:**

1. **Large-scale Pre-training:** Trained on millions of time series
2. **Zero-shot Forecasting:** No training needed for new data
3. **Uncertainty Quantification:** Provides prediction intervals
4. **Multi-horizon:** Forecasts at any horizon

**Advantages:**

- Works out-of-the-box on new time series
- No hyperparameter tuning required
- Handles various frequencies and domains
- Production-ready forecasting

## How It Works

TimeGPT represents a GPT-style architecture adapted for time series forecasting,
featuring large-scale pre-training on diverse time series data with zero-shot
and few-shot forecasting capabilities.

**Reference:** Garza et al., "TimeGPT-1", 2023.
https://arxiv.org/abs/2310.03589

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeGPT(NeuralNetworkArchitecture<>,String,TimeGPTOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance using an ONNX pretrained model. |
| `TimeGPT(NeuralNetworkArchitecture<>,TimeGPTOptions<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance in native mode for training or fine-tuning. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConfidenceLevel` | Gets the confidence level for prediction intervals. |
| `IsChannelIndependent` |  |
| `NumFeatures` |  |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` |  |
| `UseConformalPrediction` | Gets whether conformal prediction is used for uncertainty. |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `CalibrateConformalPrediction(List<Tensor<>>,List<Tensor<>>)` | Calibrates the model using historical data for conformal prediction. |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors into a single tensor. |
| `CreateNewInstance` |  |
| `DenormalizeForecast(Tensor<>)` | RevIN reverse step (Kim et al. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads TimeGPT-specific configuration during deserialization. |
| `Dispose(Boolean)` | Releases resources used by the model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to key layers for efficient access. |
| `FineTune(List<Tensor<>>,List<Tensor<>>)` | Fine-tunes the model on domain-specific data. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastOnnx(Tensor<>)` | Performs ONNX-based inference for forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through the network. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward. |
| `GenerateConformalIntervals(Tensor<>,Double[])` | Generates prediction intervals using conformal prediction. |
| `GenerateQuantilePredictions(Tensor<>,Double[])` | Generates quantile predictions through dropout-based sampling. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the model layers based on configuration. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes TimeGPT-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input tensor by appending predictions. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet TimeGPT requirements. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_calibrationResiduals` | Calibration residuals for conformal prediction. |
| `_confidenceLevel` | Confidence level for prediction intervals. |
| `_contextLength` | Context length for the input sequence. |
| `_dropout` | Dropout rate. |
| `_fineTuningLearningRate` | Learning rate for fine-tuning. |
| `_fineTuningSteps` | Number of fine-tuning steps. |
| `_forecastHorizon` | Forecast horizon for predictions. |
| `_hiddenDimension` | Hidden dimension size. |
| `_inputEmbedding` | Reference to the input embedding layer. |
| `_lossFunction` | The loss function used for training. |
| `_numFeatures` | Number of input features. |
| `_numHeads` | Number of attention heads. |
| `_numLayers` | Number of transformer layers. |
| `_optimizer` | The optimizer used for training/fine-tuning. |
| `_outputProjection` | Reference to the output projection layer. |
| `_revinMean` | RevIN (reversible instance normalization, Kim et al. |
| `_transformerLayers` | References to the transformer layers. |
| `_useConformalPrediction` | Whether to use conformal prediction for uncertainty. |
| `_useNativeMode` | Indicates whether the model is running in native mode (true) or ONNX mode (false). |

