---
title: "Int8Quantizer<T, TInput, TOutput>"
description: "INT8 quantizer for model optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Optimization.Quantization`

INT8 quantizer for model optimization.
Properly integrates with IFullModel architecture.

## For Beginners

Int8Quantizer provides AI safety functionality. Default values follow the original paper settings.

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

