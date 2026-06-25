---
title: "RealBasicVSROptions"
description: "Configuration options for the RealBasicVSR real-world video super-resolution model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the RealBasicVSR real-world video super-resolution model.

## For Beginners

Real videos have unpredictable quality issues (noise, blur, compression).
RealBasicVSR adds a "cleaning" step before upscaling that removes these artifacts first,
then uses the BasicVSR backbone for high-quality super-resolution.

## How It Works

RealBasicVSR (Chan et al., CVPR 2022) addresses real-world video SR through:

- Stochastic degradation scheme: random combination of blur, noise, resize, and compression

during training to handle unknown real-world degradations

- Pre-cleaning module: a lightweight network that removes noise/artifacts before the

BasicVSR backbone, preventing degradation from propagating across frames

- Dynamic refinement: the cleaning module strength adapts to the degradation level

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RealBasicVSROptions` | Initializes a new instance with default values. |
| `RealBasicVSROptions(RealBasicVSROptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CleaningModuleBlocks` | Gets or sets the number of residual blocks in the pre-cleaning module. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumFrames` | Gets or sets the number of input frames. |
| `NumResBlocks` | Gets or sets the number of residual blocks in the BasicVSR backbone. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `Variant` | Gets or sets the model variant. |
| `WarmupSteps` | Gets or sets the warmup steps. |

