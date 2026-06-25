---
title: "CoreMLExporter<T, TInput, TOutput>"
description: "Exports models to CoreML format for iOS deployment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Mobile.CoreML`

Exports models to CoreML format for iOS deployment.
Properly integrates with IFullModel architecture.

## For Beginners

CoreMLExporter provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `ExportFormat` |  |
| `FileExtension` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExportToBytes(IFullModel<,,>,ExportConfiguration)` |  |
| `ExportToCoreML(IFullModel<,,>,String,CoreMLConfiguration)` | Exports model directly to CoreML file. |

