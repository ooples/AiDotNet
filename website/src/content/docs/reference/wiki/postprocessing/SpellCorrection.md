---
title: "SpellCorrection<T>"
description: "SpellCorrection - Spell checking and correction for OCR output."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Postprocessing.Document`

SpellCorrection - Spell checking and correction for OCR output.

## For Beginners

OCR systems often misread characters, resulting in
misspelled words. This tool corrects them:

- Detects misspelled words
- Suggests corrections based on edit distance
- Uses context for better accuracy
- Handles domain-specific vocabulary

Key features:

- Edit distance-based suggestions
- Custom dictionary support
- Context-aware correction
- OCR-specific error patterns

Example usage:

## How It Works

SpellCorrection provides spell checking and automatic correction capabilities
specifically designed for OCR output, which has different error patterns than
normal typing errors.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpellCorrection` | Creates a new SpellCorrection instance with default settings. |
| `SpellCorrection(Int32)` | Creates a new SpellCorrection instance with specified max edit distance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsInverse` | Spell correction does not support inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddToDictionary(IEnumerable<String>)` | Adds multiple words to the custom dictionary. |
| `AddToDictionary(String)` | Adds a word to the custom dictionary. |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources used by the spell corrector. |
| `GetSuggestions(String,Int32)` | Gets correction suggestions for a misspelled word. |
| `IsCorrect(String)` | Checks if a word is spelled correctly. |
| `LoadDictionary(IEnumerable<String>)` | Loads a custom dictionary from words. |
| `ProcessCore(String)` | Corrects spelling errors in the text. |
| `ValidateInput(String)` | Validates the input text. |

