---
title: "SpeakerLM<T>"
description: "SpeakerLM language-model-based speaker diarization and verification model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Speaker`

SpeakerLM language-model-based speaker diarization and verification model.

## For Beginners

SpeakerLM figures out "who said what" in a conversation by treating
speaker changes like a language. Just as a language model predicts the next word, SpeakerLM
predicts the next speaker turn. It learns patterns like "after person A speaks, person B
usually responds" to improve accuracy.

**Usage:**

## How It Works

SpeakerLM (2024) applies language modeling techniques to speaker embeddings for improved
speaker diarization. It treats speaker turns as a sequence modeling problem, using a
transformer-based language model over speaker embeddings to predict who speaks when,
achieving improved DER on common benchmarks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpeakerLM(NeuralNetworkArchitecture<>,SpeakerLMOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a SpeakerLM model in native training mode. |
| `SpeakerLM(NeuralNetworkArchitecture<>,String,SpeakerLMOptions)` | Creates a SpeakerLM model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultThreshold` |  |
| `EmbeddingExtractor` |  |
| `IsOnnxMode` |  |
| `MinimumDurationSeconds` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Interfaces#ISpeakerEmbeddingExtractor{T}#AggregateEmbeddings(IReadOnlyList<Tensor<>>)` |  |
| `AiDotNet#Interfaces#ISpeakerEmbeddingExtractor{T}#NormalizeEmbedding(Tensor<>)` |  |
| `ComputeScore(Tensor<>,Tensor<>)` |  |
| `ComputeSimilarity(Tensor<>,Tensor<>)` |  |
| `Enroll(IReadOnlyList<Tensor<>>)` |  |
| `Enroll(Tensor<>)` |  |
| `ExtractEmbedding(Tensor<>)` |  |
| `ExtractEmbeddingAsync(Tensor<>,CancellationToken)` |  |
| `ExtractEmbeddings(IReadOnlyList<Tensor<>>)` |  |
| `GetThresholdForFAR(Double)` |  |
| `UpdateProfile(SpeakerProfile<>,Tensor<>)` |  |
| `Verify(Tensor<>,Tensor<>)` |  |
| `Verify(Tensor<>,Tensor<>,)` |  |
| `VerifyAsync(Tensor<>,Tensor<>,CancellationToken)` |  |
| `VerifyWithReferenceAudio(Tensor<>,Tensor<>)` |  |

