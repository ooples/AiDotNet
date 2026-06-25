---
title: "Float16Quantizer<T, TInput, TOutput>"
description: "FP16 (half-precision) quantizer for model optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Optimization.Quantization`

FP16 (half-precision) quantizer for model optimization.
Properly integrates with IFullModel architecture.

## For Beginners

Float16Quantizer provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `BitWidth` |  |
| `Mode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calibrate(IFullModel<,,>,IEnumerable<>)` |  |
| `GetScaleFactor(String)` |  |
| `GetZeroPoint(String)` |  |
| `Quantize(IFullModel<,,>,QuantizationConfiguration)` |  |

