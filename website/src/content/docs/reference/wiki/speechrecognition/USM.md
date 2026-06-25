---
title: "USM<T>"
description: "USM: Universal Speech Model for 100+ languages"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Multilingual`

USM: Universal Speech Model for 100+ languages

## For Beginners

USM (Universal Speech Model) is Google's 2B-parameter multilingual ASR model trained on 12M hours of unlabeled speech and 28B text sentences. It uses BEST-RQ for self-supervised pre-training, followed by universal speech-text pre-training that ali...

## How It Works

**References:**

- Paper: "Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages" (Zhang et al., Google, 2023)

USM (Universal Speech Model) is Google's 2B-parameter multilingual ASR model trained on 12M hours of unlabeled speech and 28B text sentences. It uses BEST-RQ for self-supervised pre-training, followed by universal speech-text pre-training that aligns speech and text representations. The model supports 100+ languages and achieves state-of-the-art on many benchmarks. Key innovation: speech-text joint pre-training enables zero-shot ASR for unseen languages.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using USM's universal speech-text encoder with CTC. |

