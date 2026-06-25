---
title: "NaturalSpeech2<T>"
description: "NaturalSpeech 2: latent diffusion model with continuous latent vectors for zero-shot speech synthesis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.CodecBased`

NaturalSpeech 2: latent diffusion model with continuous latent vectors for zero-shot speech synthesis.

## For Beginners

NaturalSpeech 2 uses a latent diffusion model to generate speech.
Unlike codec-based models that work with discrete tokens, it uses continuous latent vectors
from a neural audio codec. A diffusion model gradually removes noise from these vectors,
conditioned on text, duration, pitch, and speaker information, enabling zero-shot voice
cloning and even singing synthesis.

## How It Works

**References:**

- Paper: "NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers" (Shen et al., 2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NaturalSpeech2(NeuralNetworkArchitecture<>,NaturalSpeech2Options,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Initializes a new instance of the `NaturalSpeech2` class in native training/inference mode. |
| `NaturalSpeech2(NeuralNetworkArchitecture<>,String,NaturalSpeech2Options)` | Initializes a new instance of the `NaturalSpeech2` class in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDim` | Gets the hidden dimension size. |
| `NumFlowSteps` | Gets the number of diffusion steps used during inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` |  |
| `PostprocessAudio(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessText(String)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Synthesize(String)` | Synthesizes speech using NaturalSpeech 2's latent diffusion pipeline. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

