---
title: "MobileNetV2Network<T>"
description: "Implements the MobileNetV2 architecture for efficient mobile inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements the MobileNetV2 architecture for efficient mobile inference.

## For Beginners

MobileNetV2 is designed to be efficient on mobile devices.

Key innovations:

- Inverted Residuals: Expand → Depthwise Conv → Project (opposite of traditional bottlenecks)
- Linear Bottlenecks: No activation after the projection layer (preserves information)
- ReLU6: Activation capped at 6 for better quantization on mobile devices
- Depthwise Separable Convolutions: Much fewer parameters than standard convolutions

## How It Works

MobileNetV2 (Sandler et al., 2018) introduced the inverted residual structure with
linear bottlenecks, making it highly efficient for mobile and embedded vision applications.

Architecture overview:

Where t=expansion, c=output channels, n=repeat count, s=stride.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MobileNetV2Network` | Initializes a new instance of the `MobileNetV2Network` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumClasses` | Gets the number of output classes. |
| `WidthMultiplier` | Gets the width multiplier used by this network. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes and validates network-specific configuration data. |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `GetLayer(Int32)` | Gets the layer at the specified index. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `MobileNetV2_035(Int32,Int32)` | Initializes a new MobileNetV2 with width multiplier 0.35 (smallest). |
| `MobileNetV2_050(Int32,Int32)` | Initializes a new MobileNetV2 with width multiplier 0.5. |
| `MobileNetV2_075(Int32,Int32)` | Initializes a new MobileNetV2 with width multiplier 0.75. |
| `MobileNetV2_100(Int32,Int32)` | Initializes a new MobileNetV2 with width multiplier 1.0. |
| `MobileNetV2_130(Int32,Int32)` | Initializes a new MobileNetV2 with width multiplier 1.3. |
| `MobileNetV2_140(Int32,Int32)` | Initializes a new MobileNetV2 with width multiplier 1.4 (largest). |
| `PredictCore(Tensor<>)` |  |
| `PredictEager(Tensor<>)` | Routes inference through `Tensor{` for compiled-plan replay; `Tensor{` remains the eager fallback. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

