---
title: "DannaSep<T>"
description: "Danna-Sep music source separation model using dual-path attention networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.SourceSeparation`

Danna-Sep music source separation model using dual-path attention networks.

## For Beginners

Danna-Sep takes a mixed song and separates it into individual
instruments (vocals, drums, bass, other). It works by analyzing the audio from two
perspectives - time patterns and frequency patterns - to figure out which parts of
the mix belong to which instrument.

**Usage:**

## How It Works

Danna-Sep (2024) uses a dual-path attention network with a novel aggregation strategy
for music source separation. It processes audio in both time and frequency dimensions
using interleaved attention blocks, achieving competitive SDR scores with efficient
computation on the MUSDB18 benchmark.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DannaSep(NeuralNetworkArchitecture<>,DannaSepOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Danna-Sep model in native training mode. |
| `DannaSep(NeuralNetworkArchitecture<>,String,DannaSepOptions)` | Creates a Danna-Sep model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumStems` |  |
| `SupportedSources` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractSource(Tensor<>,String)` |  |
| `GetSourceMask(Tensor<>,String)` |  |
| `Remix(SourceSeparationResult<>,IReadOnlyDictionary<String,Double>)` |  |
| `RemoveSource(Tensor<>,String)` |  |
| `Separate(Tensor<>)` |  |
| `SeparateAsync(Tensor<>,CancellationToken)` |  |

