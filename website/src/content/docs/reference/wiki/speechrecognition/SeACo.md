---
title: "SeACo<T>"
description: "SeACo-Paraformer: hot-word customizable ASR"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.AlibabaASR`

SeACo-Paraformer: hot-word customizable ASR

## For Beginners

SeACo-Paraformer extends Paraformer with Semantic-Aware Contextual (SeACo) biasing for hot-word customization. A context encoder processes a list of bias phrases, and cross-attention between the decoder and context embeddings biases recognition to...

## How It Works

**References:**

- Paper: "SeACo-Paraformer: A Non-Autoregressive ASR System with Flexible and Effective Hot-Word Customization Ability" (An et al., Alibaba DAMO, 2023)

SeACo-Paraformer extends Paraformer with Semantic-Aware Contextual (SeACo) biasing for hot-word customization. A context encoder processes a list of bias phrases, and cross-attention between the decoder and context embeddings biases recognition toward specified terms. This enables accurate recognition of domain-specific terminology, proper nouns, and rare words without retraining. The biasing is applied at the semantic level rather than shallow fusion.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using SeACo's context-biased CIF parallel decoding. |

