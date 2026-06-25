---
title: "MOMENT<T>"
description: "MOMENT (Multi-task Optimization through Masked Encoding for Time series) foundation model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

MOMENT (Multi-task Optimization through Masked Encoding for Time series) foundation model.

## For Beginners

MOMENT is the first true multi-task time series foundation model:

**Architecture Overview:**

1. **Patch Embedding:** Divides the input into fixed-length patches (e.g., 64 steps each)
2. **RevIN:** Reversible Instance Normalization handles different scales automatically
3. **T5 Encoder:** A stack of transformer encoder layers processes the patches
4. **Task Heads:** Separate output heads for each supported task

**Multi-Task Capability:**

- **Forecasting:** Linear projection from encoder output to future values
- **Anomaly Detection:** Reconstruction-based — anomalies have high reconstruction error
- **Classification:** Pooled encoder output fed to a classification head
- **Imputation:** Masked patches reconstructed using unmasked context
- **Embedding:** Mean-pooled encoder output serves as the representation

**Key Insight:** MOMENT's pretraining uses masked reconstruction (like BERT),
which naturally supports all five tasks without task-specific pretraining.

**Model Sizes (MOMENT family):**

- MOMENT-Small: ~40M parameters
- MOMENT-Base: ~385M parameters (default)
- MOMENT-Large: ~1B+ parameters

## How It Works

MOMENT is a family of open time series foundation models from Carnegie Mellon University.
It uses a T5-based encoder-only transformer with patch-based input and RevIN to handle
five downstream tasks: forecasting, anomaly detection, classification, imputation, and
embedding generation — all from a single pretrained backbone.

**Reference:** Goswami et al., "MOMENT: A Family of Open Time-Series Foundation Models",
ICML 2024. https://arxiv.org/abs/2402.03885

**Thread Safety:** This class is NOT thread-safe. Create separate instances for concurrent usage.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MOMENT(NeuralNetworkArchitecture<>,MOMENTOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a MOMENT model in native mode for training or fine-tuning. |
| `MOMENT(NeuralNetworkArchitecture<>,String,MOMENTOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a MOMENT model using a pretrained ONNX model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentTask` |  |
| `IsChannelIndependent` |  |
| `MaxContextLength` |  |
| `MaxPredictionHorizon` |  |
| `ModelSize` |  |
| `NumFeatures` |  |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportedTasks` |  |
| `SupportsTraining` |  |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `Classify(Tensor<>,Int32)` |  |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectAnomalies(Tensor<>,Nullable<Double>)` |  |
| `Embed(Tensor<>)` |  |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastOnnx(Tensor<>)` | Runs inference using the ONNX model. |
| `ForwardEncoder(Tensor<>)` | Runs the encoder portion only (patches + transformer stack), stopping before the forecast / reconstruction / classification head. |
| `ForwardNative(Tensor<>)` | Performs the full native forward pass through the MOMENT architecture. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetOrBuildClassificationHead(Int32)` | Returns the classification head for the given class count, building it on demand. |
| `Impute(Tensor<>,Tensor<>)` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `ReconstructNative(Tensor<>)` | Reconstructs the input for anomaly detection and imputation tasks. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_classificationHeads` | Classification heads per Goswami et al. |
| `_reconstructionHead` | Reconstruction head for anomaly detection and imputation per Goswami et al. |
| `_useNativeMode` | Indicates whether this model uses native layers (true) or ONNX model (false). |

