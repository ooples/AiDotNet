---
title: "TtsPreprocessor"
description: "Preprocesses text for text-to-speech synthesis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.TextToSpeech`

Preprocesses text for text-to-speech synthesis.

## For Beginners

Before TTS can synthesize speech, we need to convert
written text into phonemes (speech sounds). This involves:

- Normalizing text (expanding abbreviations, numbers)
- Converting graphemes (letters) to phonemes (sounds)

For example: "Dr. Smith, 123 Main St." becomes phonemes like "D AH K T ER S M IH TH..."

## How It Works

This class handles text normalization and grapheme-to-phoneme (G2P) conversion
to prepare text for acoustic model input.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExpandNumbers(String)` | Expands numbers to words. |
| `NormalizeText(String)` | Normalizes text for TTS processing. |
| `NumberToWords(Int32)` | Converts a number to words. |
| `SplitIntoSentences(String)` | Splits text into sentences for chunked synthesis. |
| `TextToPhonemes(String)` | Converts text to phoneme IDs. |
| `WordToPhonemeIds(String)` | Converts a word to phoneme IDs. |

## Fields

| Field | Summary |
|:-----|:--------|
| `PadPhoneme` | Special phoneme IDs. |

