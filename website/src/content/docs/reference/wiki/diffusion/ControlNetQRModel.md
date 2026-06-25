---
title: "ControlNetQRModel<T>"
description: "ControlNet QR model specialized for embedding QR codes in generated artwork."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

ControlNet QR model specialized for embedding QR codes in generated artwork.

## For Beginners

This model generates beautiful artwork that secretly
contains a working QR code. When you scan the generated image with a QR reader,
it works as a real QR code, but the image looks like art rather than a plain barcode.

## How It Works

Specialized ControlNet fine-tuned for QR code pattern control. Trained to
embed scannable QR codes into aesthetically pleasing generated images while
maintaining QR readability.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ControlNetQRModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new ControlNet QR model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

