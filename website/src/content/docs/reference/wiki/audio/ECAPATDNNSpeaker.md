---
title: "ECAPATDNNSpeaker<T>"
description: "ECAPA-TDNN speaker verification and embedding extraction model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Speaker`

ECAPA-TDNN speaker verification and embedding extraction model.

## For Beginners

ECAPA-TDNN creates a unique "voiceprint" for any speaker. Feed it
audio of someone speaking, and it returns a compact vector (embedding) that captures the
speaker's unique voice characteristics. You can compare two embeddings to check if they
are the same person (verification) or find the closest match among enrolled speakers
(identification).

**Usage:**

## How It Works

ECAPA-TDNN (Desplanques et al., Interspeech 2020) is a state-of-the-art speaker embedding
architecture that extends x-vectors with Squeeze-Excitation blocks, multi-layer feature
aggregation, and channel- and context-dependent statistics pooling. Achieves 0.87% EER
on VoxCeleb1 test set.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ECAPATDNNSpeaker(NeuralNetworkArchitecture<>,ECAPATDNNSpeakerOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an ECAPA-TDNN speaker model in native training mode. |
| `ECAPATDNNSpeaker(NeuralNetworkArchitecture<>,String,ECAPATDNNSpeakerOptions)` | Creates an ECAPA-TDNN speaker model in ONNX inference mode. |

