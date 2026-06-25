---
title: "VITS<T>"
description: "VITS: end-to-end TTS with conditional VAE, normalizing flows, and adversarial training for parallel high-quality synthesis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.EndToEnd`

VITS: end-to-end TTS with conditional VAE, normalizing flows, and adversarial training for parallel high-quality synthesis.

## For Beginners

VITS (Variational Inference with adversarial learning for end-to-end
Text-to-Speech) was a breakthrough model that unified the entire TTS pipeline into a single
end-to-end architecture. Previous systems had separate components (text encoder, acoustic model,
vocoder) that were trained independently and could introduce errors at each boundary.
VITS combines three key techniques:
(1) A Conditional Variational Autoencoder (CVAE) that learns a latent representation of speech,
(2) Normalizing flows that transform simple distributions into complex speech distributions,
(3) Adversarial training (like GANs) that ensures the generated speech sounds natural.
During training, it uses Monotonic Alignment Search (MAS) to learn text-to-speech alignment
without external alignment tools. At inference, it generates high-quality speech in parallel
(all at once), making it fast while maintaining natural prosody.

## How It Works

**References:**

- Paper: "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech" (Kim et al., 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VITS(NeuralNetworkArchitecture<>,String,VITSOptions)` | Initializes a new instance of the `VITS` class in ONNX inference mode. |
| `VITS(NeuralNetworkArchitecture<>,VITSOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Initializes a new instance of the `VITS` class in native training/inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDim` | Gets the hidden dimension size. |
| `NumFlowSteps` | Gets the number of normalizing flow steps used to transform between prior and posterior distributions. |

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
| `Synthesize(String)` | Synthesizes speech from text using VITS' VAE + normalizing flow + HiFi-GAN decoder pipeline. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

