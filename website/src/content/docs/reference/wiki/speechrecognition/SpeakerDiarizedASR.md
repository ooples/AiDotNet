---
title: "SpeakerDiarizedASR<T>"
description: "Speaker-Diarized ASR: who-spoke-what transcription"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Specialized`

Speaker-Diarized ASR: who-spoke-what transcription

## For Beginners

Speaker-Diarized ASR jointly performs speech recognition and speaker diarization, producing timestamped transcriptions attributed to individual speakers. The model uses a shared Conformer encoder with a speaker-aware serialized output (SA-SOT) dec...

## How It Works

**References:**

- Paper: "Multi-talker ASR with Speaker-Aware Serialized Output Training" (2024)

Speaker-Diarized ASR jointly performs speech recognition and speaker diarization, producing timestamped transcriptions attributed to individual speakers. The model uses a shared Conformer encoder with a speaker-aware serialized output (SA-SOT) decoder that emits speaker change tokens alongside text tokens. Speaker embeddings from an auxiliary speaker encoder condition the decoder to distinguish between speakers. The system handles overlapping speech through multi-talker serialized output.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio with speaker attribution using SA-SOT decoding. |

