---
title: "Emotion2Vec<T>"
description: "emotion2vec universal speech emotion recognition model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Emotion`

emotion2vec universal speech emotion recognition model.

## For Beginners

emotion2vec detects emotions from speech. Feed it audio of someone
talking and it tells you what emotion they're expressing (happy, sad, angry, etc.) along
with a confidence score. It can also measure arousal (calm vs excited) and valence
(negative vs positive).

**Usage:**

## How It Works

emotion2vec (Ma et al., 2023) is a universal speech emotion representation model using
self-supervised pre-training on unlabeled speech followed by fine-tuning. It achieves
state-of-the-art results across multiple SER benchmarks with a single model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Emotion2Vec(NeuralNetworkArchitecture<>,Emotion2VecOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an emotion2vec model in native training mode. |
| `Emotion2Vec(NeuralNetworkArchitecture<>,String,Emotion2VecOptions)` | Creates an emotion2vec model in ONNX inference mode. |

