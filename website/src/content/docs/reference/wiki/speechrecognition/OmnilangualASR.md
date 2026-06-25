---
title: "OmnilangualASR<T>"
description: "Omnilingual ASR: universal multilingual speech recognition"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Multilingual`

Omnilingual ASR: universal multilingual speech recognition

## For Beginners

Omnilingual ASR builds on MMS with improved pre-training using 500k+ hours of multilingual data and a more efficient adapter mechanism. The model uses a language-agnostic encoder with lightweight language-specific projection heads. It supports sea...

## How It Works

**References:**

- Paper: "Omnilingual ASR" (Meta, 2024)

Omnilingual ASR builds on MMS with improved pre-training using 500k+ hours of multilingual data and a more efficient adapter mechanism. The model uses a language-agnostic encoder with lightweight language-specific projection heads. It supports seamless code-switching between languages within a single utterance, enabled by a shared multilingual vocabulary and joint language-identification CTC auxiliary loss.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using the omnilingual encoder with adaptive language projection. |

