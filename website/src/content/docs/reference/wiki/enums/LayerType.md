---
title: "LayerType"
description: "Specifies different types of layers used in neural networks, particularly in deep learning models."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies different types of layers used in neural networks, particularly in deep learning models.

## For Beginners

Neural networks are composed of layers of artificial neurons that process information.
Think of a neural network as an assembly line in a factory, where each layer is a workstation that 
performs a specific task on the data before passing it to the next layer.

Different layer types serve different purposes:

- Some extract features from data (like identifying edges in images)
- Some reduce the amount of data to process (making computation faster)
- Some make final decisions based on processed information

The combination of different layer types allows neural networks to learn complex patterns
and solve difficult problems in areas like image recognition, language processing, and more.

## Fields

| Field | Summary |
|:-----|:--------|
| `Convolutional` | A layer that applies filters to detect patterns in input data, commonly used for image processing. |
| `Dense` | A densely connected layer where each neuron is connected to every neuron in the previous layer (alias for FullyConnected). |
| `FullyConnected` | A layer where each neuron is connected to every neuron in the previous layer, used for complex pattern recognition. |
| `LoRA` | A layer implementing Low-Rank Adaptation for parameter-efficient fine-tuning. |
| `Pooling` | A layer that reduces the spatial dimensions of data by combining nearby values. |

