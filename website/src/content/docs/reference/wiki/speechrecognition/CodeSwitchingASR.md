---
title: "CodeSwitchingASR<T>"
description: "Code-Switching ASR: multilingual code-mixed speech recognition"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Specialized`

Code-Switching ASR: multilingual code-mixed speech recognition

## For Beginners

Code-Switching ASR handles speech that mixes two or more languages within a single utterance, common in multilingual communities. The model uses a shared multilingual Conformer encoder trained on code-switched corpora, with a unified vocabulary sp...

## How It Works

**References:**

- Paper: "Towards End-to-End Code-Switching Speech Recognition" (2023)

Code-Switching ASR handles speech that mixes two or more languages within a single utterance, common in multilingual communities. The model uses a shared multilingual Conformer encoder trained on code-switched corpora, with a unified vocabulary spanning all target languages. Language identification is performed implicitly through shared encoder representations. The CTC decoder uses a multilingual token set that can output tokens from any supported language within the same transcription.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes code-switched speech using a multilingual Conformer encoder. |

