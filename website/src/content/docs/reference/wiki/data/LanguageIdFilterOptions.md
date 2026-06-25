---
title: "LanguageIdFilterOptions"
description: "Configuration options for language identification filtering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Quality`

Configuration options for language identification filtering.

## How It Works

Uses character n-gram frequency profiles for language detection.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxProfileSize` | Maximum number of n-grams to keep in language profile. |
| `MinConfidence` | Minimum confidence score to accept detection. |
| `MinTextLength` | Minimum text length (in characters) for reliable detection. |
| `ProfileNGramSize` | N-gram size for character-level language profiles. |
| `TargetLanguages` | Target language codes to keep (e.g., "en", "fr"). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

