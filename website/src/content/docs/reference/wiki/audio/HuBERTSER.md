---
title: "HuBERTSER<T>"
description: "HuBERT-SER (HuBERT for Speech Emotion Recognition) model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Emotion`

HuBERT-SER (HuBERT for Speech Emotion Recognition) model.

## For Beginners

HuBERT-SER uses a model that first learned to understand speech
patterns from millions of hours of audio (HuBERT), then was specialized to detect
emotions in voice. It combines deep speech understanding with emotion classification.

**Usage:**

## How It Works

HuBERT-SER fine-tunes the HuBERT (Hsu et al., 2021) self-supervised speech model for
emotion recognition. HuBERT learns speech representations through masked prediction
of discrete speech units, achieving 69.7% weighted accuracy on IEMOCAP when fine-tuned.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HuBERTSER(NeuralNetworkArchitecture<>,HuBERTSEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a HuBERT-SER model in native training mode. |
| `HuBERTSER(NeuralNetworkArchitecture<>,String,HuBERTSEROptions)` | Creates a HuBERT-SER model in ONNX inference mode. |

