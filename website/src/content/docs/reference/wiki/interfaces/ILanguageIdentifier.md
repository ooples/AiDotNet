---
title: "ILanguageIdentifier<T>"
description: "Defines the contract for spoken language identification from audio."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for spoken language identification from audio.

## For Beginners

This is like having a friend who can tell you
"that's French!" or "that sounds like Mandarin!" just from hearing it.

How it works:

1. Extract acoustic features (phonemes, prosody, rhythm)
2. Compare to language models trained on many languages
3. Return the most likely language(s)

Applications:

- Call routing in multilingual call centers
- Automatic subtitle language selection
- Content moderation (filter by language)
- Multilingual speech recognition (select correct model)
- Immigration/border control voice analysis

Challenges:

- Code-switching (mixing languages mid-sentence)
- Accented speech (Spanish with American accent)
- Closely related languages (Norwegian vs Swedish)
- Short utterances (harder to identify with less audio)

## How It Works

Language Identification (LID) determines which language is being spoken
in an audio recording. This is different from speech recognition - we're
identifying the language, not transcribing the words.

## Properties

| Property | Summary |
|:-----|:--------|
| `SampleRate` | Gets the sample rate this identifier operates at. |
| `SupportedLanguages` | Gets the list of languages this model can identify. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AreSameLanguage(Tensor<>,Tensor<>)` | Checks if two audio samples are in the same language. |
| `GetLanguageDisplayName(String)` | Gets the display name for a language code. |
| `GetLanguageProbabilities(Tensor<>)` | Gets probabilities for all supported languages. |
| `GetTopLanguages(Tensor<>,Int32)` | Gets the top-N most likely languages. |
| `IdentifyLanguage(Tensor<>)` | Identifies the language spoken in audio. |
| `IdentifyLanguageSegments(Tensor<>,Int32)` | Identifies language with time segmentation (for multilingual audio). |

