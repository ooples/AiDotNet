---
title: "YuE<T>"
description: "YuE full-song music generation model with vocals and accompaniment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Generation`

YuE full-song music generation model with vocals and accompaniment.

## For Beginners

YuE is like having a virtual band that can write and perform an
entire song. You give it lyrics and a style ("pop, female vocalist, upbeat") and it
generates a complete song with singing, instruments, and production. Unlike most AI music
tools that only make short clips, YuE can create full-length songs.

**Usage:**

## How It Works

YuE (Yuan et al., 2025) generates complete songs with vocals and accompaniment from lyrics
and genre/style tags. It uses a dual-AR architecture: a lyrics-conditioned language model
generates semantic tokens, then a second stage produces acoustic tokens, enabling generation
of songs lasting several minutes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `YuE(NeuralNetworkArchitecture<>,String,YuEOptions)` | Creates a YuE model in ONNX inference mode. |
| `YuE(NeuralNetworkArchitecture<>,YuEOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a YuE model in native training mode. |

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
| `GenerateSong(String,String,Double,Nullable<Int32>)` | Generates a complete song from lyrics and style tags. |
| `GenerateSongAsync(String,String,Double,Nullable<Int32>,CancellationToken)` | Generates a complete song asynchronously. |
| `GetDefaultOptions` |  |
| `InpaintAudio(Tensor<>,Tensor<>,String,Int32,Nullable<Int32>)` |  |

