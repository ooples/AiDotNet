---
title: "EfficientNetNetwork<T>"
description: "Implements the EfficientNet architecture with compound scaling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements the EfficientNet architecture with compound scaling.

## For Beginners

EfficientNet achieves excellent accuracy while being very efficient.

Key innovations:

- Compound Scaling: Balances network width, depth, and resolution together
- MBConv blocks: Mobile Inverted Bottleneck with Squeeze-and-Excitation
- Swish activation: Smooth, self-gated activation function (x * sigmoid(x))
- Neural Architecture Search (NAS): The baseline B0 was found via automated search

The scaling philosophy: increasing only one dimension (width/depth/resolution)
quickly saturates accuracy. Compound scaling increases all three proportionally.

## How It Works

EfficientNet (Tan & Le, 2019) introduced compound scaling, which uniformly scales
network width, depth, and resolution using a principled approach. This achieves
state-of-the-art accuracy with significantly fewer parameters than previous models.

Architecture overview (EfficientNet-B0 baseline):

Where k=kernel size, c=output channels, n=num layers, s=stride.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EfficientNetNetwork` | Initializes a new instance of the `EfficientNetNetwork` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputResolution` | Gets the input resolution for this variant. |
| `NumClasses` | Gets the number of output classes. |
| `Variant` | Gets the EfficientNet variant. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes and validates network-specific configuration data. |
| `EfficientNetB0(Int32,Int32)` | Creates an EfficientNet-B0 network (baseline model). |
| `EfficientNetB1(Int32,Int32)` | Creates an EfficientNet-B1 network. |
| `EfficientNetB2(Int32,Int32)` | Creates an EfficientNet-B2 network. |
| `EfficientNetB3(Int32,Int32)` | Creates an EfficientNet-B3 network. |
| `EfficientNetB4(Int32,Int32)` | Creates an EfficientNet-B4 network. |
| `EfficientNetB5(Int32,Int32)` | Creates an EfficientNet-B5 network. |
| `EfficientNetB6(Int32,Int32)` | Creates an EfficientNet-B6 network. |
| `EfficientNetB7(Int32,Int32)` | Creates an EfficientNet-B7 network. |
| `ForTesting(Int32,Int32)` | Creates a minimal EfficientNet network optimized for fast test execution. |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `GetLayer(Int32)` | Gets the layer at the specified index. |
| `GetModelMetadata` |  |
| `GetNamedLayerActivations(Tensor<>)` | EfficientNet's stem-and-blocks pipeline expects rank-4 [B, C, H, W] per Tan & Le 2019 §3 — same constraint as `Tensor{`'s EnsureBatchForCnnTraining promotion. |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictEager(Tensor<>)` | Routes inference through `Tensor{` for compiled-plan replay; `Tensor{` remains the eager fallback. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

