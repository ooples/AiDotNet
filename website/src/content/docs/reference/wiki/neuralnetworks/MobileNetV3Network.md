---
title: "MobileNetV3Network<T>"
description: "Implements the MobileNetV3 architecture for efficient mobile inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements the MobileNetV3 architecture for efficient mobile inference.

## For Beginners

MobileNetV3 is the latest in the MobileNet family, optimized for
both accuracy and latency on mobile devices.

Key innovations over V2:

- Hard-Swish: A faster activation function that works better with quantization
- SE blocks: Helps the network learn which channels are most important
- Network search: The architecture was found using neural architecture search (NAS)
- Two variants: "Large" for higher accuracy, "Small" for extreme efficiency

## How It Works

MobileNetV3 (Howard et al., 2019) builds on MobileNetV2 with three key improvements:

1. Hard-Swish activation: x * min(max(0, x+3), 6) / 6 - computationally efficient
2. Squeeze-and-Excitation blocks: Adaptive channel weighting
3. Efficient network head: Reduced computational cost in final layers

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MobileNetV3Network` | Initializes a new instance with default settings. |
| `MobileNetV3Network(NeuralNetworkArchitecture<>,MobileNetV3Configuration,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Double,MobileNetV3Options)` | Initializes a new instance of the `MobileNetV3Network` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumClasses` | Gets the number of output classes. |
| `Variant` | Gets the MobileNetV3 variant used by this network. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `GetLayer(Int32)` | Gets the layer at the specified index. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `MobileNetV3Large(Int32,Int32)` | Creates a MobileNetV3-Large network. |
| `MobileNetV3Small(Int32,Int32)` | Creates a MobileNetV3-Small network. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

