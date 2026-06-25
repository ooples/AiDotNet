---
title: "LexicalWatermarker<T>"
description: "Text watermarker that embeds watermarks via synonym substitution patterns."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Watermarking`

Text watermarker that embeds watermarks via synonym substitution patterns.

## For Beginners

This watermarker swaps words with their synonyms in a
pattern that encodes a hidden signature. For example, choosing "large" over "big"
in specific positions. The meaning stays the same, but the pattern of synonym
choices reveals the watermark.

## How It Works

Uses a deterministic mapping from a secret key to select between synonym pairs.
Given a set of interchangeable word pairs (e.g., "big"/"large", "fast"/"quick"),
the watermark selects one synonym per pair based on the key. Detection checks
whether the observed synonym choices match the expected key-based pattern.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LexicalWatermarker(Double)` | Initializes a new lexical watermarker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectWatermark(String)` |  |
| `EvaluateText(String)` |  |

