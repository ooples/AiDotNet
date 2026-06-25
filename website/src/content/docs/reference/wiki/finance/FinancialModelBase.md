---
title: "FinancialModelBase<T>"
description: "Base class for all financial AI models, providing dual ONNX/native mode support."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Finance.Base`

Base class for all financial AI models, providing dual ONNX/native mode support.

## For Beginners

Think of this as a "foundation" class that all financial models build upon.

It provides common functionality that every financial model needs:

- Making predictions (inference)
- Training on data (learning)
- Saving/loading models (persistence)
- Computing gradients (for optimization)
- Integration with the AiDotNet ecosystem

The dual-mode design means you can choose:

- Native mode: More control, can train, uses more memory
- ONNX mode: Faster inference, pretrained, read-only

## How It Works

This abstract class provides the core infrastructure for financial models, following the
BLIP-2/RealESRGAN dual-mode pattern from the broader AiDotNet library. It supports both:

**Native Mode:** Full training capabilities using pure C# neural network layers.
Use the native constructor when you need to train models from scratch or fine-tune.

**ONNX Mode:** Fast inference using pretrained ONNX models.
Use the ONNX constructor for production deployment with pretrained models.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinancialModelBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Initializes a new instance with deferred configuration. |
| `FinancialModelBase(NeuralNetworkArchitecture<>,Int32,Int32,Int32,ILossFunction<>)` | Initializes a new instance using native mode for training and inference. |
| `FinancialModelBase(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32)` | Initializes a new instance using a pretrained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LastTrainingLoss` | Gets the last recorded training loss. |
| `NumFeatures` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `SupportsTraining` |  |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeserializeModelSpecificData(BinaryReader)` | Deserializes model-specific data. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` | Disposes resources. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>,Double[])` | Performs forecasting using native layers. |
| `ForecastOnnx(Tensor<>)` | Performs inference using the ONNX model. |
| `ForwardForTraining(Tensor<>)` | Tape-aware forward pass used by `NeuralNetworkBase.TrainWithTape`. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward pass through the native (non-ONNX) model. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeModelSpecificData(BinaryWriter)` | Serializes model-specific data. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `TrainCore(Tensor<>,Tensor<>,Tensor<>)` | Core training implementation for derived classes. |
| `ValidateConstructorArguments(Int32,Int32,Int32)` | Validates constructor arguments. |
| `ValidateInputShape(Tensor<>)` | Validates the input tensor shape. |

## Fields

| Field | Summary |
|:-----|:--------|
| `OnnxModelPath` | Path to the ONNX model file. |
| `OnnxSession` | The ONNX inference session for the model. |
| `_baseNumFeatures` | The number of input features. |
| `_basePredictionHorizon` | The model's prediction horizon. |
| `_baseSequenceLength` | The model's expected input sequence length. |
| `_baseUseNativeMode` | Indicates whether this model uses native layers (true) or ONNX model (false). |
| `_lastTrainingLoss` | Stores the last training loss for diagnostics. |
| `_lossHistory` | Loss history for training monitoring. |

