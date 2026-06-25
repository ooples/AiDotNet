---
title: "TFLiteConfiguration"
description: "Configuration for TensorFlow Lite model export."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.Mobile.TensorFlowLite`

Configuration for TensorFlow Lite model export.

## For Beginners

TFLiteConfiguration provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `EnableConstantFolding` | Gets or sets whether to enable constant folding (default: true). |
| `EnableGpuDelegate` | Gets or sets whether to enable GPU delegate (default: false). |
| `EnableOperatorFusion` | Gets or sets whether to enable operator fusion (default: true). |
| `ModelDescription` | Gets or sets the model description. |
| `ModelName` | Gets or sets the model name. |
| `NumThreads` | Gets or sets the number of threads for CPU inference (default: 4). |
| `QuantizationMode` | Gets or sets the quantization mode. |
| `TargetSpec` | Gets or sets the target specification for compatibility. |
| `UseDynamicRangeQuantization` | Gets or sets whether to use dynamic range quantization (default: false). |
| `UseIntegerOnlyQuantization` | Gets or sets whether to use integer-only quantization (default: false). |
| `UseNnapiDelegate` | Gets or sets whether to use NNAPI delegate for Android (default: false). |
| `UseQuantization` | Gets or sets whether to use post-training quantization (default: true). |
| `UseXnnpackDelegate` | Gets or sets whether to use XNNPACK delegate (default: true). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForAndroid` | Creates a configuration for Android deployment. |
| `ForCPU(Int32)` | Creates a configuration optimized for CPU inference. |
| `ForIOS` | Creates a configuration for iOS deployment. |
| `ForIntegerOnly` | Creates a configuration with full integer quantization. |
| `ToExportConfiguration` | Converts to ExportConfiguration. |

