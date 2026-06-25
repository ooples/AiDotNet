---
title: "ExportConfig"
description: "Configuration for exporting models to different formats and platforms."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.Configuration`

Configuration for exporting models to different formats and platforms.

## For Beginners

After training an AI model, you often need to export it to a specific
format depending on where it will run. Think of it like exporting a document to PDF, Word, or
plain text - same content, different format for different uses.

Export Formats:

- ONNX: Universal format that works everywhere (recommended for most cases)
- TensorRT: NVIDIA GPUs only, maximum performance on NVIDIA hardware
- CoreML: Apple devices (iPhone, iPad, Mac), optimized for Apple Silicon
- TFLite: Android devices and edge hardware, very efficient
- WASM: Run models in web browsers without plugins

When to export:

- Deploying to production servers (ONNX or TensorRT)
- Mobile apps (CoreML for iOS, TFLite for Android)
- Edge devices like Raspberry Pi (TFLite)
- Web applications (WASM)

Optimization:
Most export formats support optimization and quantization to make models smaller and faster.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for static shapes (default: 1). |
| `IncludeMetadata` | Gets or sets whether to include model metadata (default: true). |
| `ModelDescription` | Gets or sets the model description to include in metadata (optional). |
| `ModelName` | Gets or sets the model name to include in metadata (optional). |
| `ModelVersion` | Gets or sets the model version to include in metadata (optional). |
| `OptimizeModel` | Gets or sets whether to optimize the exported model (default: true). |
| `Quantization` | Gets or sets the quantization mode for export (default: None). |
| `TargetPlatform` | Gets or sets the target platform for export (default: CPU). |
| `ValidateAfterExport` | Gets or sets whether to validate the exported model (default: true). |

