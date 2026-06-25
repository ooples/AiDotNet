---
title: "TextAugmenterBase<T>"
description: "Base class for text data augmentations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Augmentation.Text`

Base class for text data augmentations.

## For Beginners

Text augmentation creates variations of text to improve
model robustness to different phrasings and writing styles. Common techniques include:

- Synonym replacement (replacing words with similar meanings)
- Random deletion (removing random words)
- Random swap (swapping word positions)
- Random insertion (adding synonyms of random words)
- Back-translation (translate to another language and back)

## How It Works

Text data is represented as an array of strings (sentences/documents).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TextAugmenterBase(Double,String)` | Initializes a new text augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LanguageCode` | Gets or sets the language code for language-specific operations. |
| `PreserveCase` | Gets or sets whether to preserve case when modifying text. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Detokenize(String[])` | Joins tokens back into text. |
| `GetParameters` |  |
| `IsStopword(String)` | Checks if a word is a stopword (common word to skip during augmentation). |
| `Tokenize(String)` | Tokenizes text into words. |

