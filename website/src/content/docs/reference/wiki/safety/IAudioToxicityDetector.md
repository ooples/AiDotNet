---
title: "IAudioToxicityDetector<T>"
description: "Interface for audio toxicity detection modules that identify harmful speech content."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Safety.Audio`

Interface for audio toxicity detection modules that identify harmful speech content.

## For Beginners

An audio toxicity detector checks if speech contains harmful
content like hate speech, threats, or harassment. It can work by converting speech
to text and analyzing it, or by directly analyzing the sound patterns of the voice.

## How It Works

Audio toxicity detectors analyze speech for harmful content through either
transcription-then-text-analysis (ASR pipeline) or direct acoustic feature analysis
of tone, prosody, and vocal characteristics associated with aggression or hostility.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetToxicityScore(Vector<>,Int32)` | Gets the toxicity score for the given audio (0.0 = safe, 1.0 = maximally toxic). |

