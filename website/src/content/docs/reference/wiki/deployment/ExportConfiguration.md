---
title: "ExportConfiguration"
description: "Configuration options for model export operations."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.Export`

Configuration options for model export operations.

## For Beginners

ExportConfiguration provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for static shapes (default: 1). |
| `CustomOperatorMappings` | Gets or sets custom operator mappings for unsupported operations. |
| `IncludeMetadata` | Gets or sets whether to include metadata in the exported model (default: true). |
| `InputShape` | Gets or sets the input shape dimensions (excluding batch dimension). |
| `ModelDescription` | Gets or sets the model description to include in metadata. |
| `ModelName` | Gets or sets the model name to include in metadata. |
| `ModelVersion` | Gets or sets the model version to include in metadata. |
| `OpsetVersion` | Gets or sets the target ONNX opset version (default: 13). |
| `OptimizeModel` | Gets or sets whether to optimize the exported model (default: true). |
| `OutputShape` | Gets or sets the output shape dimensions (excluding batch dimension). |
| `PlatformSpecificOptions` | Gets or sets additional platform-specific options. |
| `QuantizationMode` | Gets or sets the quantization mode for the exported model. |
| `TargetPlatform` | Gets or sets the target hardware platform for optimization. |
| `UseDynamicShapes` | Gets or sets whether to use dynamic input shapes (default: false). |
| `ValidateAfterExport` | Gets or sets whether to perform model validation after export (default: true). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForEdge` | Creates a default configuration for edge devices. |
| `ForMobile(QuantizationMode)` | Creates a default configuration for mobile export. |
| `ForTensorRT(Int32,Boolean)` | Creates a default configuration for TensorRT export. |

