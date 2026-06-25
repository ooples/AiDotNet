---
title: "CalibrationHelper<T, TInput, TOutput>"
description: "Helper class for calibrating quantizers using real forward passes."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Deployment.Optimization.Quantization.Calibration`

Helper class for calibrating quantizers using real forward passes.

## For Beginners

Calibration is the process of running sample data through a model
to understand the typical range of values (activations) that flow through each layer. This
information is crucial for accurate quantization.

## How It Works

**Why Real Forward Passes Matter:**

**Supported Models:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CalibrationHelper(QuantizationConfiguration)` | Initializes a new instance of CalibrationHelper. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CanRunPredictions(IFullModel<,,>)` | Checks if the model supports running predictions. |
| `CollectActivationStatistics(IFullModel<,,>,IEnumerable<>)` | Collects activation statistics by running calibration data through the model. |
| `CollectNeuralNetworkActivations(INeuralNetworkModel<>,IEnumerable<>,ActivationStatistics<>)` | Collects layer-by-layer activations from a neural network model. |
| `CollectParameterBasedEstimates(IFullModel<,,>,ActivationStatistics<>)` | Falls back to parameter-based estimation when forward passes aren't available. |
| `CollectPredictionBasedActivations(IFullModel<,,>,IEnumerable<>,ActivationStatistics<>)` | Collects activation statistics using model predictions. |
| `CollectTensorBasedActivations(INeuralNetwork<>,IEnumerable<Tensor<>>,ActivationStatistics<>)` | Collects activations from tensor-based neural networks using ForwardWithMemory. |
| `ComputeGlobalStatistics(IFullModel<,,>,ActivationStatistics<>)` | Computes global activation statistics from layer statistics. |
| `ConvertToTensor()` | Converts a sample to Tensor for neural network models. |
| `ExtractValuesFromOutput()` | Extracts double values from model output. |
| `FillFromParameters(Vector<>,ActivationStatistics<>)` | Fills global statistics from parameter magnitudes. |

## Fields

| Field | Summary |
|:-----|:--------|
| `MaxAcceptableFailureRate` | Maximum acceptable failure rate during calibration before warnings are triggered. |

