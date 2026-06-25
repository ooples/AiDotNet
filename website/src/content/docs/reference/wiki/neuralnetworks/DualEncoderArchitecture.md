---
title: "DualEncoderArchitecture<T>"
description: "Modality-agnostic base for two-encoder neural network architectures."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NeuralNetworks`

Modality-agnostic base for two-encoder neural network architectures.

## How It Works

Dual-encoder networks (CLIP, ALIGN, CLAP, ImageBind, …) train two parallel
encoders that produce embeddings into a shared space for contrastive
learning. The flat `Layers` list
describes only one graph, so when callers want to customise both encoders
they must reach for a richer architecture descriptor — this abstract base
holds the two layer stacks under modality-neutral names so consumers can
program against the generic dual-encoder shape, while concrete subclasses
expose semantically-named aliases (`VisionLayers`/`TextLayers`,
`AudioLayers`/`TextLayers`, etc.) for the call-site clarity each
modality pair deserves.

SOLID rationale:

- **SRP**: each concrete subclass describes exactly one

modality pairing.

- **OCP**: new modality pairs add new subclasses without

touching existing CLIP-family or audio-text consumers.

- **LSP**: tape-training and parameter-iteration code

that walks `EncoderALayers` + `EncoderBLayers` works against any
subclass.

- **ISP**: modality-specific aliases let consumers

depend only on the names that apply to them — vision-language models never
see `AudioLayers`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DualEncoderArchitecture(IEnumerable<ILayer<>>,IEnumerable<ILayer<>>,InputType,NeuralNetworkTaskType,NetworkComplexity,Int32,Int32,Int32,Int32,Int32,Boolean,Int32,Int32,Int32)` | Initializes a new dual-encoder architecture with the given encoder stacks. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EncoderALayers` | Gets the layer stack for the first encoder stream. |
| `EncoderBLayers` | Gets the layer stack for the second encoder stream. |

