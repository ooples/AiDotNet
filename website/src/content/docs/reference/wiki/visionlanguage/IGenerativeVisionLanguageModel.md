---
title: "IGenerativeVisionLanguageModel<T>"
description: "Interface for generative vision-language models that produce text output from visual input."
section: "API Reference"
---

`Interfaces` · `AiDotNet.VisionLanguage.Interfaces`

Interface for generative vision-language models that produce text output from visual input.

## How It Works

Generative VLMs take an image (and optionally a text prompt) and produce text output
such as captions, answers to visual questions, or descriptions. Architectures include:

- Q-Former bridges (BLIP-2, InstructBLIP) - lightweight adapter between frozen encoders
- Encoder-decoder (GIT, CoCa, PaLI) - ViT encoder + autoregressive text decoder
- Perceiver resampler (Flamingo, IDEFICS) - latent queries cross-attend to vision features
- Causal multimodal (KOSMOS) - visual tokens embedded directly in causal LM
- Unified generation (Emu) - single model for understanding + generation

## Properties

| Property | Summary |
|:-----|:--------|
| `DecoderEmbeddingDim` | Gets the dimensionality of the decoder embedding space. |
| `MaxGenerationLength` | Gets the maximum number of tokens the model can generate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates output token logits from an image, optionally conditioned on a text prompt. |

