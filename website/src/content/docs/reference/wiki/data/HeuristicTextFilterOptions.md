---
title: "HeuristicTextFilterOptions"
description: "Configuration options for heuristic text quality filtering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Quality`

Configuration options for heuristic text quality filtering.

## How It Works

Applies simple rule-based filters commonly used for web-crawl cleanup (e.g., C4, CCNet).

## Properties

| Property | Summary |
|:-----|:--------|
| `FilterBoilerplate` | Whether to filter documents containing common boilerplate phrases. |
| `MaxAvgWordLength` | Maximum average word length in characters. |
| `MaxDigitRatio` | Maximum ratio of digits to total characters. |
| `MaxEllipsisLineRatio` | Maximum ratio of lines that end with an ellipsis. |
| `MaxSpecialCharRatio` | Maximum ratio of special characters to total characters. |
| `MaxUppercaseRatio` | Maximum ratio of uppercase characters to alphabetic characters. |
| `MaxWordCount` | Maximum number of words in a document. |
| `MinAvgWordLength` | Minimum average word length in characters. |
| `MinPunctuationEndRatio` | Minimum proportion of lines that end with punctuation. |
| `MinWordCount` | Minimum number of words in a document. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

