---
title: "SarvamASR<T>"
description: "Sarvam AI: Indian language ASR"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.ProprietaryAPI`

Sarvam AI: Indian language ASR

## For Beginners

Sarvam AI provides speech recognition optimized for Indian languages, covering Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, and other major Indian languages. The models are trained on large-scale Indian speech corpora with...

## How It Works

**References:**

- API: "Sarvam AI Speech-to-Text" (Sarvam AI, 2024)

Sarvam AI provides speech recognition optimized for Indian languages, covering Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, and other major Indian languages. The models are trained on large-scale Indian speech corpora with coverage of regional accents, code-mixing (Hindi-English, Hinglish), and diverse recording conditions. The system uses a Conformer encoder with language-specific vocabulary and LM rescoring. Sarvam achieves significantly better accuracy on Indian languages than general multilingual ASR systems.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Sarvam AI's India-optimized ASR architecture. |

