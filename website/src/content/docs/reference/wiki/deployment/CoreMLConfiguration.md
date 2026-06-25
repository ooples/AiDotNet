---
title: "CoreMLConfiguration"
description: "Configuration for CoreML model export."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.Mobile.CoreML`

Configuration for CoreML model export.

## For Beginners

CoreMLConfiguration provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `ComputeUnits` | Gets or sets the compute units to use (CPU, GPU, Neural Engine). |
| `FlexibleInputShapes` | Gets or sets whether to enable flexible input shapes (default: false). |
| `InputFeatures` | Gets or sets input feature names and descriptions. |
| `MinimumDeploymentTarget` | Gets or sets the minimum deployment target iOS version. |
| `ModelAuthor` | Gets or sets the model author. |
| `ModelDescription` | Gets or sets the model description. |
| `ModelLicense` | Gets or sets the model license. |
| `ModelName` | Gets or sets the model name. |
| `OptimizeForSize` | Gets or sets whether to optimize for size (default: true). |
| `OutputFeatures` | Gets or sets output feature names and descriptions. |
| `QuantizationBits` | Gets or sets the quantization bits (8 or 16, default: 8). |
| `SpecVersion` | Gets or sets the CoreML specification version (default: 4). |
| `UseQuantization` | Gets or sets whether to use quantization (default: true for mobile). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForIPad` | Creates a configuration optimized for iPad. |
| `ForIPhone` | Creates a configuration optimized for iPhone. |
| `ForNeuralEngine` | Creates a configuration for Neural Engine optimization. |
| `ToExportConfiguration` | Converts to ExportConfiguration. |

