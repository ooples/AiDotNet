---
title: "Wav2Vec2ASR<T>"
description: "Wav2Vec 2.0 ASR: Meta's self-supervised speech representation model"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Foundation`

Wav2Vec 2.0 ASR: Meta's self-supervised speech representation model

## For Beginners

Wav2Vec 2.0 learns speech representations through self-supervised contrastive learning. A CNN feature encoder processes raw audio, then a Transformer encoder learns contextualized representations. During pre-training, latent speech representations...

## How It Works

**References:**

- Paper: "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (Baevski et al., Meta, 2020)

Wav2Vec 2.0 learns speech representations through self-supervised contrastive learning. A CNN feature encoder processes raw audio, then a Transformer encoder learns contextualized representations. During pre-training, latent speech representations are quantized and the model solves a contrastive task over masked positions. For ASR, a CTC head is added and the model is fine-tuned on labeled data, achieving strong results with as little as 10 minutes of labeled speech.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using wav2vec 2.0's SSL encoder with CTC decoding. |

