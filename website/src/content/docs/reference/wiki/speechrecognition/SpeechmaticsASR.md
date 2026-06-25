---
title: "SpeechmaticsASR<T>"
description: "Speechmatics: real-time multilingual ASR platform"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.ProprietaryAPI`

Speechmatics: real-time multilingual ASR platform

## For Beginners

Speechmatics provides real-time multilingual ASR optimized for accuracy across diverse audio conditions. The platform uses a proprietary Conformer-based architecture trained on diverse multilingual data. Key features include Global English (a sing...

## How It Works

**References:**

- API: "Speechmatics Real-Time ASR" (Speechmatics, 2024)

Speechmatics provides real-time multilingual ASR optimized for accuracy across diverse audio conditions. The platform uses a proprietary Conformer-based architecture trained on diverse multilingual data. Key features include Global English (a single model for all English accents), real-time and batch processing, custom dictionary for domain terms, and translation. Speechmatics supports 50+ languages with automatic language identification and handles challenging audio conditions including telephony, broadcast, and meetings.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Speechmatics' proprietary Conformer architecture. |

