---
title: "Data2VecASR<T>"
description: "Data2Vec ASR: general-purpose self-supervised learning"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Foundation`

Data2Vec ASR: general-purpose self-supervised learning

## For Beginners

Data2Vec uses a shared self-supervised framework across speech, vision, and language. The teacher network produces contextualized target representations from unmasked input, while the student network predicts these targets from masked input. Unlik...

## How It Works

**References:**

- Paper: "data2vec: A General Framework for Self-Supervised Learning in Speech, Vision and Language" (Baevski et al., Meta, 2022)

Data2Vec uses a shared self-supervised framework across speech, vision, and language. The teacher network produces contextualized target representations from unmasked input, while the student network predicts these targets from masked input. Unlike HuBERT's discrete pseudo-labels, data2vec regresses continuous latent representations. For speech, it achieves competitive ASR performance while using a simpler training objective without quantization.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using data2vec's continuous SSL encoder with CTC decoding. |

