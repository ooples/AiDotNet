---
title: "QATTrainingHook<T>"
description: "Quantization-Aware Training (QAT) hook that applies fake quantization during training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Optimization.Quantization.Training`

Quantization-Aware Training (QAT) hook that applies fake quantization during training.
Simulates quantization effects in the forward pass while allowing gradients to flow through.

## For Beginners

QAT trains the model with quantization simulation so it learns
to be robust to low-precision inference. This hook inserts "fake quantization" operations
that quantize and immediately dequantize values, simulating the precision loss.

## How It Works

**How It Works:**

**Key Components:**

**Reference:** Jacob et al., "Quantization and Training of Neural Networks for
Efficient Integer-Arithmetic-Only Inference" (2018)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QATTrainingHook(QuantizationConfiguration)` | Initializes a new instance of the QATTrainingHook. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentEpoch` | Gets the current epoch number. |
| `IsQuantizationEnabled` | Gets whether quantization is currently enabled. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyFakeQuantization(Vector<>,String)` | Applies fake quantization to weights during the forward pass. |
| `ApplyFakeQuantizationToActivations(Vector<>,String)` | Applies fake quantization to activations during the forward pass. |
| `FakeQuantize(Vector<>,QuantizationState)` | Applies fake quantization to a vector. |
| `GetLayerState(String)` | Gets the current quantization state for a layer. |
| `InitializeLayerState(Vector<>,String,Nullable<Int32>)` | Initializes quantization state for a layer. |
| `OnEpochStart(Int32)` | Called at the start of each training epoch to manage warmup and quantization state. |
| `Reset` | Resets the quantization state for all layers. |
| `StraightThroughEstimator(Vector<>,Vector<>,Vector<>)` | Applies the Straight-Through Estimator for gradient computation. |
| `UpdateActivationStatistics(Vector<>,QuantizationState)` | Updates activation statistics using exponential moving average. |
| `UpdateScales(String,Double)` | Updates quantization scales based on observed statistics (for learnable scales). |

