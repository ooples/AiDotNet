---
title: "CopyrightDetectorBase<T>"
description: "Abstract base class for copyright and memorization detection modules."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Text`

Abstract base class for copyright and memorization detection modules.

## For Beginners

This base class provides common code for all copyright detectors.
Each detector type extends this and adds its own way of checking whether an AI
is copying from copyrighted content.

## How It Works

Provides shared infrastructure for copyright detectors including n-gram extraction
and common scoring utilities. Concrete implementations provide the actual detection
algorithm (n-gram overlap, embedding similarity, perplexity analysis).

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractNgrams(String,Int32)` | Extracts character n-grams from text for overlap comparison. |
| `GetMemorizationScore(String)` |  |

