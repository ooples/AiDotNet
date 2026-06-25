---
title: "AWQQuantizer<T, TInput, TOutput>"
description: "AWQ (Activation-aware Weight Quantization) - protects important weights based on activation magnitudes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Optimization.Quantization.Strategies`

AWQ (Activation-aware Weight Quantization) - protects important weights based on activation magnitudes.
Particularly effective for very large models (70B+ parameters).

## For Beginners

AWQ observes which weights are "activated" most strongly during
inference and protects those from aggressive quantization. It's like knowing which roads
are most traveled and keeping those in better condition.

## How It Works

**How It Works:**

**Key Features:**

**Reference:** Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
Compression and Acceleration" (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AWQQuantizer(QuantizationConfiguration)` | Initializes a new instance of the AWQQuantizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BitWidth` |  |
| `IsCalibrated` | Gets whether the quantizer has been calibrated. |
| `Mode` |  |
| `UsedRealForwardPasses` | Gets whether calibration used real forward passes through the model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calibrate(IFullModel<,,>,IEnumerable<>)` |  |
| `ComputeActivationScalesFromStats` | Computes activation-based importance scales from the collected activation statistics. |
| `ComputeProtectionMask(Double[],Double[],Double)` | Computes which weights should be protected based on activation importance. |
| `FindOptimalScale(Double[],Double[],Boolean[],Int32,Int32,Int32,QuantizationConfiguration)` | Finds the optimal AWQ scaling factor using grid search. |
| `GetActivationScales(Int32)` | Gets activation scales for quantization. |
| `GetScaleFactor(String)` |  |
| `GetZeroPoint(String)` |  |
| `Quantize(IFullModel<,,>,QuantizationConfiguration)` |  |
| `QuantizeWithAWQ(Vector<>,QuantizationConfiguration)` | Quantizes parameters using the AWQ algorithm with activation-aware scaling. |

