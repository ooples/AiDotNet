---
title: "IQuantizer<T, TInput, TOutput>"
description: "Interface for model quantization strategies."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Deployment.Optimization.Quantization`

Interface for model quantization strategies.
Properly integrates with AiDotNet's IFullModel architecture.

## Properties

| Property | Summary |
|:-----|:--------|
| `BitWidth` | Gets the target bit width for quantization. |
| `Mode` | Gets the quantization mode (Int8, Float16, etc.). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calibrate(IFullModel<,,>,IEnumerable<>)` | Calibrates the quantizer using calibration data by running forward passes through the model. |
| `GetScaleFactor(String)` | Gets the scale factor for a specific layer or parameter. |
| `GetZeroPoint(String)` | Gets the zero point for a specific layer or parameter. |
| `Quantize(IFullModel<,,>,QuantizationConfiguration)` | Quantizes the model parameters using IFullModel architecture. |

