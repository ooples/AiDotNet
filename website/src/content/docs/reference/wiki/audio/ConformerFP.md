---
title: "ConformerFP<T>"
description: "Conformer-based audio fingerprinting model combining self-attention with convolutions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Fingerprinting`

Conformer-based audio fingerprinting model combining self-attention with convolutions.

## For Beginners

ConformerFP uses a powerful architecture that combines two ways of
understanding audio: attention (looking at the big picture) and convolutions (looking at
local details). This combination makes it very good at creating fingerprints that can
identify songs even from noisy or distorted recordings.

**Usage:**

## How It Works

ConformerFP applies the Conformer architecture (convolution-augmented Transformer) to audio
fingerprinting. It combines self-attention for global context with convolutions for local
feature extraction, producing highly robust fingerprints for large-scale audio retrieval.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConformerFP(NeuralNetworkArchitecture<>,ConformerFPOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a ConformerFP model in native training mode. |
| `ConformerFP(NeuralNetworkArchitecture<>,String,ConformerFPOptions)` | Creates a ConformerFP model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FingerprintLength` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSimilarity(AudioFingerprint<>,AudioFingerprint<>)` |  |
| `FindMatches(AudioFingerprint<>,AudioFingerprint<>,Int32)` |  |
| `Fingerprint(Tensor<>)` |  |
| `Fingerprint(Vector<>)` |  |

