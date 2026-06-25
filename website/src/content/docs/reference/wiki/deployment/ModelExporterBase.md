---
title: "ModelExporterBase<T, TInput, TOutput>"
description: "Abstract base class for model exporters that provides common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Deployment.Export`

Abstract base class for model exporters that provides common functionality.
Properly integrates with IFullModel architecture.

## For Beginners

for provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `ExportFormat` |  |
| `FileExtension` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CanExport(IFullModel<,,>)` |  |
| `Export(IFullModel<,,>,String,ExportConfiguration)` |  |
| `ExportToBytes(IFullModel<,,>,ExportConfiguration)` |  |
| `GetInputShape(IFullModel<,,>,ExportConfiguration)` | Gets the input shape from the model or configuration. |
| `GetLayerSummary(IFullModel<,,>)` | Gets a summary of all layers for export metadata, using `ILayeredModel` when available. |
| `GetOutputShape(IFullModel<,,>,ExportConfiguration)` | Gets the output shape from the model or configuration. |
| `GetValidationErrors(IFullModel<,,>)` |  |
| `ValidateExportedModel(String,ExportConfiguration)` | Validates the exported model file. |

