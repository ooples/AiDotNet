---
title: "TripleStreamArchitecture<T>"
description: "A neural network architecture for three-stream models (e.g., perceiver/abstractor/resampler-style generative VLMs and cross-modality fusion encoders) that hosts a vision encoder, an auxiliary stream (perceiver / abstractor / resampler / cro…"
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

A neural network architecture for three-stream models (e.g., perceiver/abstractor/resampler-style
generative VLMs and cross-modality fusion encoders) that hosts a vision encoder, an auxiliary
stream (perceiver / abstractor / resampler / cross-modality fusion), and a text or decoder stream.

## For Beginners

Bigger vision-language models have three pieces — a vision tower, an
adapter or "fusion" tower in the middle, and a text or language-decoder tower — that pass data
to each other in sequence. If you want to swap in custom layers for all three, this architecture
lets you describe each piece independently instead of jamming them into one list (which drops the
last two pieces).

## How It Works

Models like IDEFICS2, MPLUGOwl3, Qwen3VL, BridgeTower, LXMERT, and METER need three independent
layer stacks. The flat `Layers` property cannot describe
three graphs, so callers wanting custom layers for all streams should use this type. The
`VisionLayers` populates the model's primary `Layers` list, while `AuxiliaryLayers`
and `TextOrDecoderLayers` populate the model-specific auxiliary lists (the perceiver /
abstractor / resampler / cross-modality stack and the language decoder / text encoder
respectively).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TripleStreamArchitecture(IEnumerable<ILayer<>>,IEnumerable<ILayer<>>,IEnumerable<ILayer<>>,InputType,NeuralNetworkTaskType,NetworkComplexity,Int32,Int32,Int32,Int32,Int32,Boolean,Int32,Int32,Int32)` | Initializes a new triple-stream architecture with explicit vision, auxiliary, and text/decoder layers. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLayers` | Gets the auxiliary stream layer stack (perceiver / abstractor / resampler / cross-modality fusion / bridge depending on model). |
| `TextOrDecoderLayers` | Gets the layer stack for the text encoder or language decoder, depending on model. |
| `VisionLayers` | Gets the layer stack for the vision encoder. |

