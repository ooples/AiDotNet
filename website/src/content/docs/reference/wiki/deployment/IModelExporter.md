---
title: "IModelExporter<T, TInput, TOutput>"
description: "Base interface for model exporters that convert AiDotNet models to various deployment formats."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Deployment.Export`

Base interface for model exporters that convert AiDotNet models to various deployment formats.
Properly integrates with IFullModel architecture.

## Properties

| Property | Summary |
|:-----|:--------|
| `ExportFormat` | Gets the target export format (e.g., "ONNX", "TensorFlowLite", "CoreML", "TensorRT") |
| `FileExtension` | Gets the file extension for the exported model (e.g., ".onnx", ".tflite", ".mlmodel") |

## Methods

| Method | Summary |
|:-----|:--------|
| `CanExport(IFullModel<,,>)` | Validates that the model can be exported to this format. |
| `Export(IFullModel<,,>,String,ExportConfiguration)` | Exports the model to the specified path with the given configuration. |
| `ExportToBytes(IFullModel<,,>,ExportConfiguration)` | Exports the model to a byte array with the given configuration. |
| `GetValidationErrors(IFullModel<,,>)` | Gets validation errors if the model cannot be exported. |

