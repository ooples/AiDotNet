---
title: "LanguageIdFilter"
description: "Filters documents based on detected language using character n-gram profiles."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Quality`

Filters documents based on detected language using character n-gram profiles.

## How It Works

Uses a simple but effective character n-gram frequency approach for language detection.
Requires training with reference text for each target language.
Based on the Cavnar-Trenkle (1994) text categorization method.

## Methods

| Method | Summary |
|:-----|:--------|
| `AddLanguageProfile(String,IReadOnlyList<String>)` | Adds a language profile built from reference text. |
| `DetectLanguage(String)` | Detects the most likely language of a text. |
| `Filter(IReadOnlyList<String>)` | Filters documents by language, returning indices of documents that should be removed. |

