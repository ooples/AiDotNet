---
title: "MusicGenModel<T>"
description: "Meta's MusicGen model for generating music from text descriptions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.MusicGen`

Meta's MusicGen model for generating music from text descriptions.

## For Beginners

MusicGen creates original music from your descriptions:

How it works:

1. You describe the music you want ("upbeat jazz piano")
2. The text encoder understands your description
3. The language model generates a sequence of "music tokens"
4. The EnCodec decoder converts tokens to actual audio

Key features:

- 30 seconds of high-quality 32kHz audio
- Multiple genres and styles
- Control over instruments, tempo, mood
- Stereo output option

Usage:

## How It Works

MusicGen is a state-of-the-art text-to-music generation model from Meta AI Research.
It uses a single-stage transformer language model that operates directly on EnCodec
audio codes, generating high-quality music from text descriptions.

Architecture components:

- **Text Encoder:** T5-based encoder that converts text prompts to embeddings
- **Language Model:** Transformer decoder that generates audio codes autoregressively
- **EnCodec Decoder:** Neural audio codec that converts discrete codes to waveforms

Reference: "Simple and Controllable Music Generation" by Copet et al., Meta AI, 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MusicGenModel` | Creates a MusicGen model with default configuration for native training. |
| `MusicGenModel(NeuralNetworkArchitecture<>,MusicGenOptions,ITokenizer,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a MusicGen model using native layers for training from scratch. |
| `MusicGenModel(NeuralNetworkArchitecture<>,String,String,String,ITokenizer,MusicGenOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a MusicGen model using pretrained ONNX models for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxDurationSeconds` | Gets the maximum duration of audio that can be generated. |
| `SampleRate` | Gets the sample rate of generated audio. |
| `SupportsAudioContinuation` | Gets whether this model supports audio continuation. |
| `SupportsAudioInpainting` | Gets whether this model supports audio inpainting. |
| `SupportsTextToAudio` | Gets whether this model supports text-to-audio generation. |
| `SupportsTextToMusic` | Gets whether this model supports text-to-music generation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ContinueAudio(Tensor<>,String,Double,Int32,Nullable<Int32>)` | Continues existing audio by extending it. |
| `CreateNewInstance` | Creates a new instance for cloning. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data. |
| `Dispose(Boolean)` | Disposes of resources. |
| `GenerateAudio(String,String,Double,Int32,Double,Nullable<Int32>)` | Generates audio from a text description. |
| `GenerateAudioAsync(String,String,Double,Int32,Double,Nullable<Int32>,CancellationToken)` | Generates audio asynchronously. |
| `GenerateMusic(String,String,Double,Int32,Double,Nullable<Int32>)` | Generates music from a text description. |
| `GetDefaultOptions` | Gets default generation options. |
| `GetModelMetadata` | Gets model metadata. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers following the golden standard pattern. |
| `InpaintAudio(Tensor<>,Tensor<>,String,Int32,Nullable<Int32>)` | Inpainting is not supported by MusicGen. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output. |
| `PredictCore(Tensor<>)` | Makes a prediction using the model. |
| `PreprocessAudio(Tensor<>)` | Preprocesses raw audio for model input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on input data. |
| `UpdateParameters(Vector<>)` | Updates model parameters. |

