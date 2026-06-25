---
title: "UniTS<T>"
description: "UniTS (Unified Time Series Model) implementation for multi-task time series processing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

UniTS (Unified Time Series Model) implementation for multi-task time series processing.

## For Beginners

UniTS is designed to be a universal time series model:

**The Key Insight:**
Different time series tasks share common patterns. Instead of training
separate models, UniTS learns a unified representation that works for all tasks.

**Supported Tasks:**

1. **Forecasting:** Predict future values
2. **Classification:** Categorize entire time series
3. **Anomaly Detection:** Identify unusual patterns
4. **Imputation:** Fill in missing values

**Architecture:**

- Multi-scale temporal convolution for local patterns (different kernel sizes)
- Transformer layers for global dependencies
- Task-specific output heads for different outputs
- Shared backbone pretrained on diverse datasets

**Advantages:**

- One model for multiple tasks (transfer learning)
- Strong zero-shot performance on new domains
- Efficient inference (shared computation)

## How It Works

UniTS is a unified architecture that handles multiple time series tasks including
forecasting, classification, anomaly detection, and imputation using a single
pretrained model with task-specific output heads.

**Reference:** Gao et al., "UniTS: A Unified Multi-Task Time Series Model", 2024.
https://arxiv.org/abs/2403.00131

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UniTS(NeuralNetworkArchitecture<>,String,UniTSOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance using an ONNX pretrained model. |
| `UniTS(NeuralNetworkArchitecture<>,UniTSOptions<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance in native mode for training. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConvKernelSizes` | Gets the convolution kernel sizes. |
| `IsChannelIndependent` |  |
| `NumClasses` | Gets the number of classes for classification task. |
| `NumFeatures` |  |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` |  |
| `TaskType` | Gets the task type (forecasting, classification, anomaly, imputation). |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `ApplySoftmax(Tensor<>)` | Applies softmax normalization to convert logits to probabilities. |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `Classify(Tensor<>)` | Performs classification by computing class probabilities. |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors into a single tensor. |
| `CreateNewInstance` |  |
| `DenormalizeForecast(Tensor<>)` | RevIN reverse step (Kim et al. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads UniTS-specific configuration during deserialization. |
| `DetectAnomalies(Tensor<>)` | Performs anomaly detection by computing reconstruction error. |
| `Dispose(Boolean)` | Releases resources used by the model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to key layers for efficient access. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastOnnx(Tensor<>)` | Performs ONNX-based inference for task-specific predictions. |
| `Forward(Tensor<>)` | Performs the forward pass through the network. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward. |
| `GenerateQuantilePredictions(Tensor<>,Double[])` | Generates quantile predictions through Monte Carlo dropout. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `Impute(Tensor<>,Tensor<>)` | Performs imputation by filling missing values. |
| `InitializeLayers` | Initializes the model layers based on configuration. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes UniTS-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input tensor by appending predictions. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet UniTS requirements. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_contextLength` | Context length for the input sequence. |
| `_convKernelSizes` | Convolution kernel sizes for multi-scale processing. |
| `_dropout` | Dropout rate for regularization. |
| `_forecastHorizon` | Forecast horizon for predictions. |
| `_hiddenDimension` | Hidden dimension size. |
| `_inputEmbedding` | Reference to the input embedding layer. |
| `_lossFunction` | The loss function used for training. |
| `_multiScaleLayers` | References to the multi-scale processing layers. |
| `_numClasses` | Number of classes for classification task. |
| `_numFeatures` | Number of input features. |
| `_numHeads` | Number of attention heads. |
| `_numLayers` | Number of transformer layers. |
| `_optimizer` | The optimizer used for training. |
| `_outputProjection` | Reference to the output projection layer. |
| `_revinMean` | RevIN (reversible instance normalization, Kim et al. |
| `_taskType` | Task type (forecasting, classification, anomaly, imputation). |
| `_transformerLayers` | References to the transformer attention layers. |
| `_useNativeMode` | Indicates whether the model is running in native mode (true) or ONNX mode (false). |

