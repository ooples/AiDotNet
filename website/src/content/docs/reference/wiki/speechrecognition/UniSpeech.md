---
title: "UniSpeech<T>"
description: "UniSpeech: unified pre-training for speech representation learning"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Foundation`

UniSpeech: unified pre-training for speech representation learning

## For Beginners

UniSpeech unifies self-supervised and supervised pre-training in a single framework. The model jointly optimizes a contrastive self-supervised loss and a CTC loss when labeled data is available. This multi-task pre-training allows the model to lev...

## How It Works

**References:**

- Paper: "UniSpeech: Unified Pre-training for Self-Supervised Learning and Supervised Learning for ASR" (Wang et al., Microsoft, 2021)

UniSpeech unifies self-supervised and supervised pre-training in a single framework. The model jointly optimizes a contrastive self-supervised loss and a CTC loss when labeled data is available. This multi-task pre-training allows the model to leverage both unlabeled and labeled speech data during pre-training, producing representations that are more aligned with the downstream ASR task than purely self-supervised approaches.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using UniSpeech's unified SSL/supervised encoder with CTC. |

