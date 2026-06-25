---
title: "IDetectionBackbone<T>"
description: "Marker interface for detection backbones."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Marker interface for detection backbones. Combines the two contracts a detection
model expects from its backbone:

- `INeuralNetworkModel` — standard neural-network surface

(Layers, GetNamedLayerActivations, UpdateParameters, state-dict serialization).

- `IFeatureMapProvider` — multi-scale feature pyramid output

(GetFeatureMaps, OutputChannels, Strides) consumed by FPN, anchor generators,
and DETR-style transformer heads.

## For Beginners

A detection model needs two things from its "backbone"
(the network that processes the image first): the standard ability to train and run
like any other neural network, and a special ability to output features at multiple
scales for spotting both small and large objects. This interface bundles those two
requirements so detection models can ask for "anything that can do both" rather than
"specifically a BackboneBase".

## How It Works

Use this interface as the field type in detection consumers (ObjectDetectorBase,
TextDetectorBase) instead of the concrete `BackboneBase<T>` base class.
That keeps detection models decoupled from a specific base-class hierarchy and lets
any future backbone implementation that satisfies the two underlying contracts plug
in without inheriting from `BackboneBase`.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractFeatures(Tensor<>)` | Extracts multi-scale features from an input image tensor. |
| `GetParameterCount` | Returns the total parameter count across the backbone's internal layers. |
| `ReadParameters(BinaryReader)` | Reads the backbone's complete parameter state from a binary reader. |
| `WriteParameters(BinaryWriter)` | Writes the backbone's complete parameter state to a binary writer for persistence. |

