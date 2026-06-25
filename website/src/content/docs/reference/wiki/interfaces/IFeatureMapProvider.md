---
title: "IFeatureMapProvider<T>"
description: "Mixin interface for neural networks that produce multi-scale feature pyramids — typically detection / segmentation backbones (ResNet, CSPDarknet, EfficientNet, SwinTransformer) whose outputs feed FPN, PAN, anchor generators, or DETR-style t…"
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Mixin interface for neural networks that produce multi-scale feature pyramids — typically
detection / segmentation backbones (ResNet, CSPDarknet, EfficientNet, SwinTransformer)
whose outputs feed FPN, PAN, anchor generators, or DETR-style transformer heads.

## For Beginners

Detection models (YOLO, FasterRCNN, DETR) need to look at an image
at multiple zoom levels — small features for tiny objects, large features for big objects.
A "feature pyramid" provides exactly that: the same network outputs feature maps at several
resolutions. This interface is the contract any model implements when it wants to plug into
a detection head as a feature pyramid source.

## How It Works

Replaces the legacy `BackboneBase<T>.ExtractFeatures()` contract. By implementing
this interface alongside `INeuralNetworkModel`, a backbone gains the standard
neural-network surface (Layers, GetNamedLayerActivations, UpdateParameters, SetTrainingMode,
state-dict serialization) AND keeps the multi-scale extraction API that detection heads need.

The contract is paper-faithful: `Tensor{` returns one tensor per scale in
resolution-descending order (high-resolution first), with corresponding entries in
`OutputChannels` and `Strides`. For ResNet-50 detection use, that's
typically `{ C2, C3, C4, C5 }` with channels `{ 256, 512, 1024, 2048 }` and
strides `{ 4, 8, 16, 32 }`.

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` | The number of channels in each scale's feature map. |
| `Strides` | The downsampling stride at each scale (input pixel count divided by feature pixel count). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFeatureMaps(Tensor<>)` | Runs the network on `input` and returns the multi-scale feature maps, in resolution-descending order (the highest-resolution map first). |

