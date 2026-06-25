---
title: "CLAP<T>"
description: "CLAP (Contrastive Language-Audio Pre-training) model for zero-shot and fine-tuned audio classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Classification`

CLAP (Contrastive Language-Audio Pre-training) model for zero-shot and fine-tuned audio classification.

## For Beginners

CLAP is special because it understands both audio and text. Instead of having
fixed labels like "dog bark" or "siren", you can describe any sound in plain English and CLAP will
find it in audio. For example, you can search for "the sound of rain hitting a tin roof" without
ever training on that specific label. This is called "zero-shot" classification.

**Usage:**

## How It Works

CLAP (Wu et al., ICASSP 2023) learns joint audio-text representations through contrastive learning,
similar to CLIP for images. It enables zero-shot audio classification by comparing audio embeddings
with text descriptions, achieving 26.7% zero-shot accuracy on ESC-50 and 46.8% mAP on AudioSet
with fine-tuning.

**Architecture:** CLAP consists of two encoders:

- **Audio encoder**: HTS-AT or PANN backbone processing 64-bin mel spectrograms at 48 kHz
- **Text encoder**: RoBERTa-based encoder for natural language sound descriptions
- **Projection heads**: Map both modalities into a shared 512-dim embedding space
- **Contrastive loss**: InfoNCE with learnable temperature aligns matching audio-text pairs

**References:**

- Paper: "Large-Scale Contrastive Language-Audio Pre-Training with Feature Fusion and Keyword-to-Caption Augmentation" (Wu et al., ICASSP 2023)
- Repository: https://github.com/LAION-AI/CLAP

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CLAP(NeuralNetworkArchitecture<>,CLAPOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a CLAP model in native training mode for training from scratch. |
| `CLAP(NeuralNetworkArchitecture<>,String,CLAPOptions)` | Creates a CLAP model in ONNX inference mode from a pre-trained audio encoder model file. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EventLabels` |  |
| `SupportedEvents` |  |
| `TimeResolution` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateAsync(CLAPOptions,IProgress<Double>,CancellationToken)` | Downloads and creates a CLAP model asynchronously from a model repository. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Detect(Tensor<>)` |  |
| `Detect(Tensor<>,)` |  |
| `DetectAsync(Tensor<>,CancellationToken)` |  |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>)` |  |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>,)` |  |
| `Dispose(Boolean)` |  |
| `ExtractAudioEmbedding(Tensor<>)` | Extracts audio embeddings from the audio encoder. |
| `GetEventProbabilities(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessAudio(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SetTextPrompts(String[])` | Sets the text prompts for zero-shot classification. |
| `StartStreamingSession` |  |
| `StartStreamingSession(Int32,)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

