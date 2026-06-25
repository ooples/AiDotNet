---
title: "W2vBERT<T>"
description: "W2v-BERT: combining contrastive learning and masked language modeling for speech"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Foundation`

W2v-BERT: combining contrastive learning and masked language modeling for speech

## For Beginners

W2v-BERT combines the contrastive learning objective of wav2vec 2.0 with the masked prediction objective of BERT. The lower Transformer layers solve a contrastive task to produce discrete tokens, while upper layers perform masked language modeling...

## How It Works

**References:**

- Paper: "w2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training" (Chung et al., Google, 2021)

W2v-BERT combines the contrastive learning objective of wav2vec 2.0 with the masked prediction objective of BERT. The lower Transformer layers solve a contrastive task to produce discrete tokens, while upper layers perform masked language modeling over these tokens. This two-stage approach within a single model produces representations that capture both acoustic and linguistic information, achieving strong ASR performance.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using w2v-BERT's dual-objective SSL encoder with CTC. |

