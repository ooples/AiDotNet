---
title: "HuBERTASR<T>"
description: "HuBERT ASR: Hidden-Unit BERT for self-supervised speech representation"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Foundation`

HuBERT ASR: Hidden-Unit BERT for self-supervised speech representation

## For Beginners

HuBERT uses an offline clustering step to provide pseudo-labels for masked prediction pre-training. Unlike wav2vec 2.0's contrastive loss, HuBERT predicts discrete cluster assignments of masked speech segments. The model iteratively refines its cl...

## How It Works

**References:**

- Paper: "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units" (Hsu et al., Meta, 2021)

HuBERT uses an offline clustering step to provide pseudo-labels for masked prediction pre-training. Unlike wav2vec 2.0's contrastive loss, HuBERT predicts discrete cluster assignments of masked speech segments. The model iteratively refines its clusters using learned representations from previous iterations. This approach produces rich speech representations that transfer well to ASR, speaker verification, and speech emotion recognition tasks.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using HuBERT's masked prediction encoder with CTC decoding. |

