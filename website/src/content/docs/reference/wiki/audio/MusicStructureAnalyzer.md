---
title: "MusicStructureAnalyzer<T>"
description: "Music Structure Analyzer that segments songs into structural sections (intro, verse, chorus, etc.)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.MusicAnalysis`

Music Structure Analyzer that segments songs into structural sections (intro, verse, chorus, etc.).

## For Beginners

This model listens to a song and identifies its sections - where the
verse begins, where the chorus kicks in, and where the bridge or outro happens. It's like
creating an automatic table of contents for a song.

**Usage:**

## How It Works

The Music Structure Analyzer segments songs into structural sections (intro, verse, chorus,
bridge, outro) using a neural network trained on annotated music datasets. It combines
self-similarity matrix features with a segmentation network.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MusicStructureAnalyzer(NeuralNetworkArchitecture<>,MusicStructureAnalyzerOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Music Structure Analyzer in native training mode. |
| `MusicStructureAnalyzer(NeuralNetworkArchitecture<>,String,MusicStructureAnalyzerOptions)` | Creates a Music Structure Analyzer in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumSections` | Gets the number of structural sections. |
| `SectionLabels` | Gets the section labels this model can detect. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnalyzeStructure(Tensor<>)` | Analyzes the structure of a song, returning labeled time segments. |
| `AnalyzeStructureAsync(Tensor<>,CancellationToken)` | Analyzes music structure asynchronously. |
| `GetSectionProbabilities(Tensor<>)` | Gets per-frame section probabilities. |

