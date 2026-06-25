---
title: "TFLiteExporter<T, TInput, TOutput>"
description: "Exports models to TensorFlow Lite format for mobile deployment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Mobile.TensorFlowLite`

Exports models to TensorFlow Lite format for mobile deployment.
Properly integrates with IFullModel architecture.

## For Beginners

TFLiteExporter provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `ExportFormat` |  |
| `FileExtension` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExportToBytes(IFullModel<,,>,ExportConfiguration)` |  |
| `ExportToTFLite(IFullModel<,,>,String,TFLiteConfiguration)` | Exports model directly to TFLite file with specific configuration. |

