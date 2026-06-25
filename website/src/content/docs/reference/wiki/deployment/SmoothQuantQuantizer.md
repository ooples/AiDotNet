---
title: "SmoothQuantQuantizer<T, TInput, TOutput>"
description: "SmoothQuant - transfers quantization difficulty from activations to weights using per-channel smoothing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Optimization.Quantization.Strategies`

SmoothQuant - transfers quantization difficulty from activations to weights using per-channel smoothing.
Enables efficient W8A8 quantization (both weights and activations at 8-bit).

## For Beginners

Activations (intermediate values during inference) often have
outliers that are very hard to quantize. SmoothQuant "smooths" these outliers by mathematically
transferring some of their range to the weights, making both easier to quantize.

## How It Works

**How It Works:**

**Key Features:**

**Reference:** Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training
Quantization for Large Language Models" (2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SmoothQuantQuantizer(QuantizationConfiguration)` | Initializes a new instance of the SmoothQuantQuantizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BitWidth` |  |
| `IsCalibrated` | Gets whether the quantizer has been calibrated. |
| `Mode` |  |
| `SmoothingScales` | Gets the smoothing scales applied to transform weights. |
| `UsedRealForwardPasses` | Gets whether calibration used real forward passes through the model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calibrate(IFullModel<,,>,IEnumerable<>)` |  |
| `ComputeAsymmetricZeroPoint(Double,Double,Double)` | Computes the zero-point for asymmetric quantization. |
| `ComputeSmoothingScales(Double[],Double[])` | Computes smoothing scales using the SmoothQuant formula. |
| `GetScaleFactor(String)` |  |
| `GetSmoothingScales(Int32)` | Gets smoothing scales for the given size. |
| `GetZeroPoint(String)` |  |
| `Quantize(IFullModel<,,>,QuantizationConfiguration)` |  |
| `QuantizePerChannel(Double[],[],Int32,QuantizationConfiguration,Double,Double)` | Performs per-channel quantization. |
| `QuantizePerGroup(Double[],[],Int32,QuantizationConfiguration,Double,Double)` | Performs per-group quantization. |
| `QuantizeWithSmoothQuant(Vector<>,QuantizationConfiguration)` | Quantizes parameters using SmoothQuant with per-channel smoothing. |

