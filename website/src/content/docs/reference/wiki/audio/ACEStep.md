---
title: "ACEStep<T>"
description: "ACE-Step accelerated consistency-enhanced music generation model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Generation`

ACE-Step accelerated consistency-enhanced music generation model.

## For Beginners

ACE-Step generates music from text descriptions super fast. While
most AI music generators need many steps (like painting layer by layer), ACE-Step can
create music in just 1-4 steps, making it fast enough for real-time use. You describe
the music you want and it creates it almost instantly.

**Usage:**

## How It Works

ACE-Step (2024) uses consistency training to generate high-quality music in just 1-4
diffusion steps instead of the 50-100 steps needed by standard models. It achieves
real-time music generation while maintaining quality comparable to multi-step models,
making it practical for interactive applications.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ACEStep(NeuralNetworkArchitecture<>,ACEStepOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an ACE-Step model in native training mode. |
| `ACEStep(NeuralNetworkArchitecture<>,String,ACEStepOptions)` | Creates an ACE-Step model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsOnnxMode` |  |
| `MaxDurationSeconds` |  |
| `SupportsAudioContinuation` |  |
| `SupportsAudioInpainting` |  |
| `SupportsTextToAudio` |  |
| `SupportsTextToMusic` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ContinueAudio(Tensor<>,String,Double,Int32,Nullable<Int32>)` |  |
| `GenerateAudio(String,String,Double,Int32,Double,Nullable<Int32>)` |  |
| `GenerateAudioAsync(String,String,Double,Int32,Double,Nullable<Int32>,CancellationToken)` |  |
| `GenerateMusic(String,String,Double,Int32,Double,Nullable<Int32>)` |  |
| `GetDefaultOptions` |  |
| `InpaintAudio(Tensor<>,Tensor<>,String,Int32,Nullable<Int32>)` |  |

