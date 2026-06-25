---
title: "GPTQQuantizer<T, TInput, TOutput>"
description: "GPTQ (Generative Pre-trained Transformer Quantization) - state-of-the-art weight quantization using second-order Hessian information to minimize reconstruction error."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Optimization.Quantization.Strategies`

GPTQ (Generative Pre-trained Transformer Quantization) - state-of-the-art weight quantization
using second-order Hessian information to minimize reconstruction error.

## For Beginners

GPTQ is like a smart packing algorithm that knows which items
are most important. It uses advanced math (Hessian matrix) to figure out which weights
matter most and handles those more carefully during compression.

## How It Works

**How It Works:**

**Key Features:**

**Reference:** Frantar et al., "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers" (2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GPTQQuantizer(QuantizationConfiguration)` | Initializes a new instance of the GPTQQuantizer. |

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
| `ComputeHessianFromActivationStats` | Computes Hessian approximation from activation statistics collected via forward passes. |
| `ComputeHessianFromParameters(Vector<>)` | Computes Hessian approximation directly from parameters (fallback). |
| `GetHessianCrossElement(Int32,Int32,Double[])` | Gets Hessian cross-element (off-diagonal) approximation. |
| `GetHessianDiagonal(Int32,Int32)` | Gets the Hessian diagonal for a specific range. |
| `GetHessianDiagonalValue(Int32)` | Gets the Hessian diagonal value for a single global index. |
| `GetProcessingOrder(Double[],Int32,Boolean)` | Gets the processing order for columns based on activation importance (ActOrder optimization). |
| `GetScaleFactor(String)` |  |
| `GetZeroPoint(String)` |  |
| `Quantize(IFullModel<,,>,QuantizationConfiguration)` |  |
| `QuantizeWithGPTQ(Vector<>,QuantizationConfiguration)` | Quantizes parameters using the GPTQ algorithm with Hessian-based error compensation. |

