---
title: "RevAI<T>"
description: "Rev AI: human-quality transcription API"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.ProprietaryAPI`

Rev AI: human-quality transcription API

## For Beginners

Rev AI provides human-quality automatic transcription trained on Rev.com's millions of hours of human-transcribed audio. The models benefit from Rev's unique position as both a human transcription service and AI company, using human-generated labe...

## How It Works

**References:**

- API: "Rev AI" (Rev.com, 2024)

Rev AI provides human-quality automatic transcription trained on Rev.com's millions of hours of human-transcribed audio. The models benefit from Rev's unique position as both a human transcription service and AI company, using human-generated labels for training. Features include speaker diarization, custom vocabulary, profanity filtering, and real-time streaming. Rev AI supports multiple languages and achieves near-human accuracy on conversational English through continual training on human-corrected transcriptions.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Rev AI's human-trained ASR architecture. |

