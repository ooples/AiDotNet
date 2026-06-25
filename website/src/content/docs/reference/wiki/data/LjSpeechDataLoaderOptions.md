---
title: "LjSpeechDataLoaderOptions"
description: "Configuration options for the LJSpeech 1.1 data loader (Ito & Johnson 2017)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Audio.Benchmarks`

Configuration options for the LJSpeech 1.1 data loader (Ito & Johnson 2017).

## How It Works

LJSpeech is the canonical single-speaker English TTS corpus — 13,100
audiobook clips (≈ 24 hours) from a single female narrator at 22,050 Hz.
Used as the default TTS training corpus by Tacotron, FastSpeech, VITS,
and most subsequent neural TTS papers.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxAudioSamples` | Maximum waveform length in samples (zero-padded). |
| `MaxTextLength` | Maximum transcript length in tokens (input text). |
| `Split` | Dataset split. |
| `UseNormalizedText` | Use the normalized text (numbers/dates expanded). |
| `VocabularySize` | Vocabulary size for the text tokenizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

